// Package v1alpha1 contains API Schema definitions for the densecore v1alpha1 API group.
// DenseCoreModel CRD는 Kubernetes에서 LLM 모델을 선언적으로 배포하고 관리합니다.
// +kubebuilder:object:generate=true
// +groupName=densecore.io
package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ============================================================================
// DenseCoreModel Types
// ============================================================================

// DenseCoreModelSpec defines the desired state of DenseCoreModel.
// DenseCoreModel의 원하는 상태를 정의합니다.
type DenseCoreModelSpec struct {
	// Model specifies the model to deploy
	// +kubebuilder:validation:Required
	Model ModelSpec `json:"model"`

	// Replicas is the number of DenseCore server pods
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	// +optional
	Replicas *int32 `json:"replicas,omitempty"`

	// Resources defines the resource requirements for the container
	// +optional
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`

	// Autoscaling configures horizontal pod autoscaling
	// +optional
	Autoscaling *AutoscalingSpec `json:"autoscaling,omitempty"`

	// Service configures the Kubernetes Service
	// +optional
	Service ServiceSpec `json:"service,omitempty"`

	// Env is additional environment variables for the container
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// NodeSelector is a selector which must be true for the pod to fit on a node
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Tolerations are the pod's tolerations
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// Affinity is the pod's scheduling constraints
	// +optional
	Affinity *corev1.Affinity `json:"affinity,omitempty"`

	// Image overrides the default DenseCore image
	// +optional
	Image string `json:"image,omitempty"`

	// ImagePullPolicy specifies the image pull policy
	// +kubebuilder:validation:Enum=Always;IfNotPresent;Never
	// +kubebuilder:default=IfNotPresent
	// +optional
	ImagePullPolicy corev1.PullPolicy `json:"imagePullPolicy,omitempty"`
}

// ModelSpec defines the model source configuration.
// 모델 소스 설정을 정의합니다.
type ModelSpec struct {
	// RepoID is the HuggingFace repository ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
	// +kubebuilder:validation:Required
	RepoID string `json:"repoId"`

	// Filename is the model filename within the repository
	// +kubebuilder:validation:Required
	Filename string `json:"filename"`

	// HFTokenSecretRef is a reference to a secret containing the HuggingFace token
	// +optional
	HFTokenSecretRef *SecretKeySelector `json:"hfTokenSecretRef,omitempty"`

	// Threads is the number of inference threads
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=4
	// +optional
	Threads int32 `json:"threads,omitempty"`
}

// SecretKeySelector selects a key from a Secret.
type SecretKeySelector struct {
	// Name is the name of the secret
	Name string `json:"name"`

	// Key is the key in the secret to use
	// +kubebuilder:default=HF_TOKEN
	Key string `json:"key,omitempty"`
}

// AutoscalingSpec configures horizontal pod autoscaling.
// 수평 Pod 자동 스케일링을 설정합니다.
type AutoscalingSpec struct {
	// Enabled enables autoscaling
	// +kubebuilder:default=false
	Enabled bool `json:"enabled"`

	// MinReplicas is the minimum number of replicas
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	MinReplicas int32 `json:"minReplicas,omitempty"`

	// MaxReplicas is the maximum number of replicas
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=5
	MaxReplicas int32 `json:"maxReplicas,omitempty"`

	// TargetCPUUtilization is the target CPU utilization percentage
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=70
	// +optional
	TargetCPUUtilization int32 `json:"targetCPUUtilization,omitempty"`

	// TargetMemoryUtilization is the target memory utilization percentage
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100
	// +optional
	TargetMemoryUtilization *int32 `json:"targetMemoryUtilization,omitempty"`
}

// ServiceSpec configures the Kubernetes Service.
// Kubernetes 서비스를 설정합니다.
type ServiceSpec struct {
	// GRPCEnabled enables the gRPC server
	// +kubebuilder:default=true
	// +optional
	GRPCEnabled *bool `json:"grpcEnabled,omitempty"`

	// Type is the Kubernetes Service type
	// +kubebuilder:validation:Enum=ClusterIP;LoadBalancer;NodePort
	// +kubebuilder:default=ClusterIP
	// +optional
	Type corev1.ServiceType `json:"type,omitempty"`

	// Port is the service port
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:default=8080
	// +optional
	Port int32 `json:"port,omitempty"`

	// GRPCPort is the gRPC service port
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:default=50051
	// +optional
	GRPCPort int32 `json:"grpcPort,omitempty"`

	// Annotations are additional service annotations
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`
}

// DenseCoreModelStatus defines the observed state of DenseCoreModel.
// DenseCoreModel의 관찰된 상태를 정의합니다.
type DenseCoreModelStatus struct {
	// Phase is the current phase of the DenseCoreModel
	// +kubebuilder:validation:Enum=Pending;Downloading;Loading;Running;Failed
	Phase string `json:"phase,omitempty"`

	// ReadyReplicas is the number of ready replicas
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// Conditions represent the current conditions of the DenseCoreModel
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LoadProgress is the model loading progress (0-100)
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	// +optional
	LoadProgress int32 `json:"loadProgress,omitempty"`

	// Endpoint is the service endpoint URL
	// +optional
	Endpoint string `json:"endpoint,omitempty"`

	// GRPCEndpoint is the gRPC service endpoint
	// +optional
	GRPCEndpoint string `json:"grpcEndpoint,omitempty"`

	// LastUpdated is the timestamp of the last status update
	// +optional
	LastUpdated *metav1.Time `json:"lastUpdated,omitempty"`

	// Message provides additional information about the current state
	// +optional
	Message string `json:"message,omitempty"`

	// ObservedGeneration is the generation observed by the controller
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// ============================================================================
// DenseCoreModel Resource
// ============================================================================

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.readyReplicas
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase",description="Current phase"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas",description="Ready replicas"
// +kubebuilder:printcolumn:name="Endpoint",type="string",JSONPath=".status.endpoint",description="Service endpoint"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName=dcm

// DenseCoreModel is the Schema for the densecoremodels API.
// DenseCoreModel은 densecoremodels API의 스키마입니다.
type DenseCoreModel struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DenseCoreModelSpec   `json:"spec,omitempty"`
	Status DenseCoreModelStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DenseCoreModelList contains a list of DenseCoreModel.
type DenseCoreModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DenseCoreModel `json:"items"`
}

// ============================================================================
// Phase Constants
// ============================================================================

const (
	// PhasePending indicates the model is pending deployment
	PhasePending = "Pending"

	// PhaseDownloading indicates the model is being downloaded
	PhaseDownloading = "Downloading"

	// PhaseLoading indicates the model is being loaded into memory
	PhaseLoading = "Loading"

	// PhaseRunning indicates the model is running and ready
	PhaseRunning = "Running"

	// PhaseFailed indicates the model deployment failed
	PhaseFailed = "Failed"
)

// ============================================================================
// Condition Types
// ============================================================================

const (
	// ConditionTypeReady indicates the model is ready to serve requests
	ConditionTypeReady = "Ready"

	// ConditionTypeModelLoaded indicates the model has been loaded
	ConditionTypeModelLoaded = "ModelLoaded"

	// ConditionTypeProgressing indicates the deployment is progressing
	ConditionTypeProgressing = "Progressing"

	// ConditionTypeDegraded indicates the deployment is degraded
	ConditionTypeDegraded = "Degraded"
)

func init() {
	SchemeBuilder.Register(&DenseCoreModel{}, &DenseCoreModelList{})
}
