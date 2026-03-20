package grpc

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"testing"
	"time"
)

func TestBuildServerTLSCredentialsRequiresCertAndKey(t *testing.T) {
	_, err := buildServerTLSCredentials(Config{TLSEnabled: true})
	if err == nil {
		t.Fatal("expected error for missing cert/key")
	}
}

func TestBuildServerTLSCredentialsRequiresCAForMTLS(t *testing.T) {
	certFile, keyFile := writeSelfSignedCertPair(t)
	defer func() {
		_ = os.Remove(certFile)
		_ = os.Remove(keyFile)
	}()

	_, err := buildServerTLSCredentials(Config{
		TLSEnabled:           true,
		TLSCertFile:          certFile,
		TLSKeyFile:           keyFile,
		TLSRequireClientCert: true,
		TLSClientCAFile:      "",
	})
	if err == nil {
		t.Fatal("expected error for missing client CA file")
	}
}

func writeSelfSignedCertPair(t *testing.T) (string, string) {
	t.Helper()

	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("failed to generate key: %v", err)
	}

	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: "densecore-test",
		},
		NotBefore:             time.Now().Add(-time.Minute),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              []string{"localhost"},
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
	if err != nil {
		t.Fatalf("failed to create cert: %v", err)
	}

	certPem := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPem := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})

	certF, err := os.CreateTemp(t.TempDir(), "grpc-cert-*.pem")
	if err != nil {
		t.Fatalf("failed to create cert temp file: %v", err)
	}
	keyF, err := os.CreateTemp(t.TempDir(), "grpc-key-*.pem")
	if err != nil {
		t.Fatalf("failed to create key temp file: %v", err)
	}

	if _, err := certF.Write(certPem); err != nil {
		t.Fatalf("failed to write cert file: %v", err)
	}
	if _, err := keyF.Write(keyPem); err != nil {
		t.Fatalf("failed to write key file: %v", err)
	}

	_ = certF.Close()
	_ = keyF.Close()

	return certF.Name(), keyF.Name()
}
