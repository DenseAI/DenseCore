package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/server"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"
)

var runCmd = &cobra.Command{
	Use:   "run [model_name]",
	Short: "Run a model with interactive chat",
	Long:  `Run a model directly in your terminal with a beautiful chat interface.`,
	Args:  cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelName := "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
		if len(args) > 0 {
			modelName = args[0]
		}

		filename, _ := cmd.Flags().GetString("filename")
		port, _ := cmd.Flags().GetInt("port")
		if port == 0 {
			port = 8080
		}

		// Run the TUI program
		p := tea.NewProgram(initialModel(modelName, filename, port), tea.WithAltScreen())
		if _, err := p.Run(); err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
	},
}

// ============================================================================
// TUI Styles
// ============================================================================

var (
	titleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#04B575")).
			Bold(true).
			Padding(0, 1)

	subtitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#666666")).
			Italic(true)

	statusStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#aaaaaa")).
			Padding(1, 2)

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff5555")).
			Bold(true).
			Padding(1, 2)

	userMsgStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#61afef")).
			Bold(true)

	assistantMsgStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#98c379"))

	progressBarFilled = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#04B575"))

	progressBarEmpty = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#444444"))

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#626262"))
)

// ============================================================================
// ProgressReader
// ============================================================================

type ProgressReader struct {
	reader     io.Reader
	total      int64
	current    int64
	lastUpdate time.Time
	onProgress func(downloaded, total int64)
	mu         sync.Mutex
}

func (pr *ProgressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	if n > 0 {
		pr.mu.Lock()
		pr.current += int64(n)
		if time.Since(pr.lastUpdate) > 100*time.Millisecond {
			pr.lastUpdate = time.Now()
			if pr.onProgress != nil {
				pr.onProgress(pr.current, pr.total)
			}
		}
		pr.mu.Unlock()
	}
	return n, err
}

// ============================================================================
// Model & State
// ============================================================================

type state int

const (
	stateChecking state = iota
	stateDownloading
	stateStarting
	stateChat
	stateError
)

type chatMessage struct {
	Role    string
	Content string
}

type model struct {
	modelName       string
	filename        string
	port            int
	state           state
	progress        float64
	downloadedBytes int64
	totalBytes      int64
	errorMsg        string

	messages      []chatMessage
	textInput     textinput.Model
	isStreaming   bool
	streamingText string

	// Channels for async operations
	streamChan   chan streamEvent
	downloadChan chan tea.Msg

	// Server instance for graceful shutdown
	serverInstance *server.ServerInstance

	// UI components
	spinner       spinner.Model
	width, height int
}

type streamEvent struct {
	token string
	done  bool
	err   error
}

func initialModel(modelName, filename string, port int) model {
	if filename == "" {
		filename = inferFilename(modelName)
	}

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#04B575"))

	ti := textinput.New()
	ti.Placeholder = "Type a message..."
	ti.CharLimit = 2000
	ti.Width = 60

	return model{
		modelName:    modelName,
		filename:     filename,
		port:         port,
		state:        stateChecking,
		messages:     []chatMessage{},
		streamChan:   make(chan streamEvent, 100),
		downloadChan: make(chan tea.Msg, 100),
		spinner:      s,
		textInput:    ti,
	}
}

func inferFilename(modelName string) string {
	lowerName := strings.ToLower(modelName)
	if strings.Contains(lowerName, "qwen") {
		return "qwen2.5-0.5b-instruct-q4_k_m.gguf"
	}
	if strings.Contains(lowerName, "llama-2-7b") {
		return "llama-2-7b-chat.Q4_K_M.gguf"
	}
	return "model.gguf"
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		tea.WindowSize(),
		m.spinner.Tick,
		checkModelCmd(m.filename),
	)
}

// ============================================================================
// Messages
// ============================================================================

type modelExistsMsg struct{ path string }
type modelNotFoundMsg struct{}
type downloadProgressMsg struct {
	downloaded int64
	total      int64
}
type downloadCompleteMsg struct{ path string }
type downloadErrorMsg struct{ err error }
type serverReadyMsg struct {
	port     int
	instance *server.ServerInstance
}
type serverErrorMsg struct{ err error }
type streamTokenMsg struct{ token string }
type streamDoneMsg struct{}
type streamErrorMsg struct{ err error }
type portFoundMsg struct{ port int }

func closeQuietly(closer io.Closer) {
	_ = closer.Close()
}

// ============================================================================
// Commands
// ============================================================================

func checkModelCmd(filename string) tea.Cmd {
	return func() tea.Msg {
		cacheDir := getModelCacheDir()
		modelPath := filepath.Join(cacheDir, filename)
		if _, err := os.Stat(modelPath); err == nil {
			return modelExistsMsg{path: modelPath}
		}
		return modelNotFoundMsg{}
	}
}

// startDownloadCmd starts download in goroutine and sends events to channel
func startDownloadCmd(modelName, filename string, sub chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		go func() {
			cacheDir := getModelCacheDir()
			if err := os.MkdirAll(cacheDir, 0755); err != nil {
				sub <- downloadErrorMsg{err: err}
				return
			}

			modelPath := filepath.Join(cacheDir, filename)
			url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", modelName, filename)

			resp, err := http.Get(url)
			if err != nil {
				sub <- downloadErrorMsg{err: err}
				return
			}
			defer closeQuietly(resp.Body)

			if resp.StatusCode != http.StatusOK {
				sub <- downloadErrorMsg{err: fmt.Errorf("HTTP %d", resp.StatusCode)}
				return
			}

			file, err := os.Create(modelPath)
			if err != nil {
				sub <- downloadErrorMsg{err: err}
				return
			}

			// ProgressReader sends updates through the channel
			pr := &ProgressReader{
				reader:     resp.Body,
				total:      resp.ContentLength,
				lastUpdate: time.Now(),
				onProgress: func(curr, total int64) {
					sub <- downloadProgressMsg{downloaded: curr, total: total}
				},
			}

			if _, err := io.Copy(file, pr); err != nil {
				closeQuietly(file)
				sub <- downloadErrorMsg{err: err}
				return
			}
			if err := file.Close(); err != nil {
				sub <- downloadErrorMsg{err: err}
				return
			}

			sub <- downloadCompleteMsg{path: modelPath}
		}()
		return nil
	}
}

// waitForDownloadMsg waits for next message from download channel
func waitForDownloadMsg(sub chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		return <-sub
	}
}

// findAvailablePortCmd finds an available port
func findAvailablePortCmd(startPort int) tea.Cmd {
	return func() tea.Msg {
		for port := startPort; port < startPort+100; port++ {
			addr := fmt.Sprintf("127.0.0.1:%d", port)
			listener, err := net.Listen("tcp", addr)
			if err == nil {
				closeQuietly(listener)
				return portFoundMsg{port: port}
			}
		}
		return serverErrorMsg{err: fmt.Errorf("no available ports found in range %d-%d", startPort, startPort+100)}
	}
}

// startServerCmd starts the server and waits for health check
func startServerCmd(modelPath string, port int) tea.Cmd {
	return func() tea.Msg {
		grpcEnabled := false
		opts := &server.Options{
			Host:        "127.0.0.1",
			Port:        port,
			ModelPath:   modelPath,
			LogOutput:   io.Discard,
			ShowBanner:  false,
			Background:  true,
			GRPCEnabled: &grpcEnabled,
		}

		instance, err := server.Start(opts)
		if err != nil {
			return serverErrorMsg{err: err}
		}

		// Health Check polling
		client := &http.Client{Timeout: 1 * time.Second}
		for i := 0; i < 60; i++ { // 30초 대기
			time.Sleep(500 * time.Millisecond)
			resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health/live", port))
			if err == nil && resp.StatusCode == http.StatusOK {
				closeQuietly(resp.Body)
				return serverReadyMsg{port: port, instance: instance}
			}
			if resp != nil {
				closeQuietly(resp.Body)
			}
		}
		return serverErrorMsg{err: fmt.Errorf("timeout waiting for server")}
	}
}

// waitForStreamMsg waits for next token from stream channel
func waitForStreamMsg(sub chan streamEvent) tea.Cmd {
	return func() tea.Msg {
		evt := <-sub
		if evt.done {
			return streamDoneMsg{}
		}
		if evt.err != nil {
			return streamErrorMsg{err: evt.err}
		}
		return streamTokenMsg{token: evt.token}
	}
}

// startStreamingChat starts the HTTP request and pushes tokens to channel
func startStreamingChat(port int, messages []chatMessage, streamChan chan streamEvent) {
	go func() {
		defer func() {
			streamChan <- streamEvent{done: true}
		}()

		reqMessages := make([]map[string]string, len(messages))
		for i, msg := range messages {
			reqMessages[i] = map[string]string{
				"role":    msg.Role,
				"content": msg.Content,
			}
		}

		reqBody := map[string]interface{}{
			"model":      "densecore",
			"messages":   reqMessages,
			"max_tokens": 512,
			"stream":     true,
		}

		jsonBody, _ := json.Marshal(reqBody)

		resp, err := http.Post(
			fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
			"application/json",
			bytes.NewReader(jsonBody),
		)
		if err != nil {
			streamChan <- streamEvent{err: err}
			return
		}
		defer closeQuietly(resp.Body)

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			streamChan <- streamEvent{err: fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))}
			return
		}

		// Parse SSE stream
		reader := bufio.NewReader(resp.Body)

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				streamChan <- streamEvent{err: err}
				return
			}

			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}

			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				streamChan <- streamEvent{token: chunk.Choices[0].Delta.Content}
			}
		}
	}()
}

// ============================================================================
// Update
// ============================================================================

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.textInput.Width = m.width - 10
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "esc":
			// Graceful Shutdown
			if m.serverInstance != nil {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				defer cancel()
				_ = m.serverInstance.Shutdown(ctx)
			}
			return m, tea.Quit

		case "enter":
			if m.state == stateChat && m.textInput.Value() != "" && !m.isStreaming {
				userMsg := chatMessage{Role: "user", Content: m.textInput.Value()}
				m.messages = append(m.messages, userMsg)
				m.textInput.Reset()
				m.isStreaming = true
				m.streamingText = ""

				startStreamingChat(m.port, m.messages, m.streamChan)
				return m, waitForStreamMsg(m.streamChan)
			}
		}

		// Update text input
		if m.state == stateChat && !m.isStreaming {
			var cmd tea.Cmd
			m.textInput, cmd = m.textInput.Update(msg)
			cmds = append(cmds, cmd)
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		cmds = append(cmds, cmd)

	case modelExistsMsg:
		m.state = stateStarting
		return m, tea.Batch(m.spinner.Tick, findAvailablePortCmd(m.port))

	case modelNotFoundMsg:
		m.state = stateDownloading
		m.progress = 0
		// Start download + subscribe to channel
		return m, tea.Batch(
			startDownloadCmd(m.modelName, m.filename, m.downloadChan),
			waitForDownloadMsg(m.downloadChan),
		)

	case downloadProgressMsg:
		m.downloadedBytes = msg.downloaded
		m.totalBytes = msg.total
		if msg.total > 0 {
			m.progress = float64(msg.downloaded) / float64(msg.total)
		}
		// Wait for next download message
		return m, waitForDownloadMsg(m.downloadChan)

	case downloadCompleteMsg:
		m.progress = 1.0
		m.state = stateStarting
		return m, tea.Batch(m.spinner.Tick, findAvailablePortCmd(m.port))

	case downloadErrorMsg:
		m.state = stateError
		m.errorMsg = fmt.Sprintf("Download failed: %v", msg.err)
		return m, nil

	case portFoundMsg:
		m.port = msg.port
		modelPath := filepath.Join(getModelCacheDir(), m.filename)
		return m, tea.Batch(m.spinner.Tick, startServerCmd(modelPath, m.port))

	case serverReadyMsg:
		m.serverInstance = msg.instance
		m.port = msg.port
		m.state = stateChat
		m.textInput.Focus()
		m.messages = append(m.messages, chatMessage{
			Role:    "system",
			Content: "You are a helpful AI assistant.",
		})
		return m, textinput.Blink

	case serverErrorMsg:
		m.state = stateError
		m.errorMsg = fmt.Sprintf("Server failed: %v", msg.err)
		return m, nil

	case streamTokenMsg:
		m.streamingText += msg.token
		return m, waitForStreamMsg(m.streamChan)

	case streamDoneMsg:
		m.isStreaming = false
		if m.streamingText != "" {
			m.messages = append(m.messages, chatMessage{
				Role:    "assistant",
				Content: m.streamingText,
			})
		}
		m.streamingText = ""
		m.textInput.Focus()
		return m, textinput.Blink

	case streamErrorMsg:
		m.isStreaming = false
		m.messages = append(m.messages, chatMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("Error: %v", msg.err),
		})
		m.textInput.Focus()
		return m, textinput.Blink
	}

	return m, tea.Batch(cmds...)
}

// ============================================================================
// View
// ============================================================================

func (m model) View() string {
	var s strings.Builder

	// Header
	s.WriteString(titleStyle.Render("🚀 DenseCore AI"))
	s.WriteString("\n")
	s.WriteString(subtitleStyle.Render(m.modelName))
	s.WriteString("\n\n")

	switch m.state {
	case stateChecking:
		s.WriteString(statusStyle.Render(m.spinner.View() + " Checking for local model..."))

	case stateDownloading:
		width := 40
		filled := int(m.progress * float64(width))
		empty := width - filled

		bar := progressBarFilled.Render(strings.Repeat("█", filled)) +
			progressBarEmpty.Render(strings.Repeat("░", empty))

		sizeInfo := ""
		if m.totalBytes > 0 {
			sizeInfo = fmt.Sprintf("\n   %.1f MB / %.1f MB",
				float64(m.downloadedBytes)/1024/1024,
				float64(m.totalBytes)/1024/1024)
		}

		s.WriteString(statusStyle.Render(fmt.Sprintf("📥 Downloading model...\n\n   [%s] %.1f%%%s", bar, m.progress*100, sizeInfo)))

	case stateStarting:
		s.WriteString(statusStyle.Render(m.spinner.View() + fmt.Sprintf(" Starting inference server on port %d...\n\n   This may take a moment while the model loads.", m.port)))

	case stateChat:
		// Calculate available height for messages
		headerHeight := 4
		inputHeight := 4
		availableHeight := m.height - headerHeight - inputHeight
		if availableHeight < 5 {
			availableHeight = 5
		}

		// Show messages (skip system message)
		visibleMessages := m.messages
		if len(visibleMessages) > 0 && visibleMessages[0].Role == "system" {
			visibleMessages = visibleMessages[1:]
		}

		// Limit visible messages to fit screen
		maxMessages := availableHeight / 3
		if maxMessages < 3 {
			maxMessages = 3
		}
		if len(visibleMessages) > maxMessages {
			visibleMessages = visibleMessages[len(visibleMessages)-maxMessages:]
		}

		for _, msg := range visibleMessages {
			switch msg.Role {
			case "user":
				s.WriteString(userMsgStyle.Render("You: "))
				s.WriteString(msg.Content)
			case "assistant":
				s.WriteString(assistantMsgStyle.Render("AI: "))
				s.WriteString(msg.Content)
			}
			s.WriteString("\n\n")
		}

		// Show streaming response
		if m.isStreaming {
			s.WriteString(assistantMsgStyle.Render("AI: "))
			if m.streamingText != "" {
				s.WriteString(m.streamingText)
			}
			s.WriteString(m.spinner.View())
			s.WriteString("\n\n")
		}

		// Input area
		s.WriteString(m.textInput.View())
		s.WriteString("\n\n")
		s.WriteString(helpStyle.Render("Enter ↵ send • Ctrl+C exit • Port: " + fmt.Sprintf("%d", m.port)))

	case stateError:
		s.WriteString(errorStyle.Render("❌ " + m.errorMsg))
		s.WriteString("\n\n")
		s.WriteString(helpStyle.Render("Press Ctrl+C to exit"))
	}

	return s.String()
}

// ============================================================================
// Helpers
// ============================================================================

func getModelCacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".densecore", "models")
}

func init() {
	runCmd.Flags().StringP("filename", "f", "", "Specific GGUF filename to download")
	runCmd.Flags().IntP("port", "p", 8080, "Port for the local server")
	rootCmd.AddCommand(runCmd)
}
