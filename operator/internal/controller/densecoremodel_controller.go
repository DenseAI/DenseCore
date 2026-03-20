// Package controller contains the DenseCoreModel controller implementation.
// DenseCoreModel 컨트롤러는 CRD를 관찰하고 Deployment, Service, HPA를 생성/관리합니다.
package controller

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	densecoreioiov1alpha1 "github.com/densecore/operator/api/v1alpha1"
)

const (
	// DefaultImage is the default DenseCore server image
	DefaultImage = "densecore/densecore:0.3.0"

	// FinalizerName is the finalizer name for DenseCoreModel
	FinalizerName = "densecore.io/finalizer"

	// RequeueAfter is the default requeue interval
	RequeueAfter = 30 * time.Second
)

// DenseCoreModelReconciler reconciles a DenseCoreModel object.
// DenseCoreModel 오브젝트를 조정(reconcile)합니다.
type DenseCoreModelReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=densecore.io,resources=densecoremodels,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=densecore.io,resources=densecoremodels/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=densecore.io,resources=densecoremodels/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete

// Reconcile handles the reconciliation loop for DenseCoreModel.
// DenseCoreModel에 대한 조정 루프를 처리합니다.
func (r *DenseCoreModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Reconciling DenseCoreModel", "name", req.Name, "namespace", req.Namespace)

	// 1. Fetch the DenseCoreModel instance
	dcm := &densecoreioiov1alpha1.DenseCoreModel{}
	if err := r.Get(ctx, req.NamespacedName, dcm); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("DenseCoreModel not found, might have been deleted")
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get DenseCoreModel")
		return ctrl.Result{}, err
	}

	// 2. Handle finalizer
	if dcm.ObjectMeta.DeletionTimestamp.IsZero() {
		// Add finalizer if not present
		if !controllerutil.ContainsFinalizer(dcm, FinalizerName) {
			controllerutil.AddFinalizer(dcm, FinalizerName)
			if err := r.Update(ctx, dcm); err != nil {
				return ctrl.Result{}, err
			}
		}
	} else {
		// Handle deletion
		if controllerutil.ContainsFinalizer(dcm, FinalizerName) {
			if err := r.cleanupResources(ctx, dcm); err != nil {
				return ctrl.Result{}, err
			}
			controllerutil.RemoveFinalizer(dcm, FinalizerName)
			if err := r.Update(ctx, dcm); err != nil {
				return ctrl.Result{}, err
			}
		}
		return ctrl.Result{}, nil
	}

	// 3. Set initial status if empty
	if dcm.Status.Phase == "" {
		dcm.Status.Phase = densecoreioiov1alpha1.PhasePending
		if err := r.Status().Update(ctx, dcm); err != nil {
			return ctrl.Result{}, err
		}
	}

	// 4. Create or update Deployment
	deployment, err := r.reconcileDeployment(ctx, dcm)
	if err != nil {
		r.setFailedStatus(ctx, dcm, "DeploymentFailed", err.Error())
		return ctrl.Result{}, err
	}

	// 5. Create or update Service
	service, err := r.reconcileService(ctx, dcm)
	if err != nil {
		r.setFailedStatus(ctx, dcm, "ServiceFailed", err.Error())
		return ctrl.Result{}, err
	}

	// 6. Create or update HPA if autoscaling is enabled
	if dcm.Spec.Autoscaling != nil && dcm.Spec.Autoscaling.Enabled {
		if _, err := r.reconcileHPA(ctx, dcm); err != nil {
			r.setFailedStatus(ctx, dcm, "HPAFailed", err.Error())
			return ctrl.Result{}, err
		}
	}

	// 7. Update status
	r.updateStatus(ctx, dcm, deployment, service)

	log.Info("Reconciliation complete", "phase", dcm.Status.Phase)
	return ctrl.Result{RequeueAfter: RequeueAfter}, nil
}

// ============================================================================
// Deployment Reconciliation
// ============================================================================

func (r *DenseCoreModelReconciler) reconcileDeployment(ctx context.Context, dcm *densecoreioiov1alpha1.DenseCoreModel) (*appsv1.Deployment, error) {
	log := log.FromContext(ctx)

	deployment := &appsv1.Deployment{}
	deploymentName := types.NamespacedName{Name: dcm.Name, Namespace: dcm.Namespace}

	err := r.Get(ctx, deploymentName, deployment)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}

	desired := r.buildDeployment(dcm)
	if err := controllerutil.SetControllerReference(dcm, desired, r.Scheme); err != nil {
		return nil, err
	}

	if apierrors.IsNotFound(err) {
		log.Info("Creating Deployment", "name", desired.Name)
		if err := r.Create(ctx, desired); err != nil {
			return nil, err
		}
		return desired, nil
	}

	// Update if needed
	log.Info("Updating Deployment", "name", deployment.Name)
	deployment.Spec = desired.Spec
	if err := r.Update(ctx, deployment); err != nil {
		return nil, err
	}

	return deployment, nil
}

func (r *DenseCoreModelReconciler) buildDeployment(dcm *densecoreioiov1alpha1.DenseCoreModel) *appsv1.Deployment {
	labels := map[string]string{
		"app.kubernetes.io/name":       "densecore",
		"app.kubernetes.io/instance":   dcm.Name,
		"app.kubernetes.io/managed-by": "densecore-operator",
	}

	replicas := int32(1)
	if dcm.Spec.Replicas != nil {
		replicas = *dcm.Spec.Replicas
	}

	image := DefaultImage
	if dcm.Spec.Image != "" {
		image = dcm.Spec.Image
	}

	imagePullPolicy := corev1.PullIfNotPresent
	if dcm.Spec.ImagePullPolicy != "" {
		imagePullPolicy = dcm.Spec.ImagePullPolicy
	}

	threads := int32(4)
	if dcm.Spec.Model.Threads > 0 {
		threads = dcm.Spec.Model.Threads
	}

	grpcEnabled := true
	if dcm.Spec.Service.GRPCEnabled != nil {
		grpcEnabled = *dcm.Spec.Service.GRPCEnabled
	}

	httpPort := int32(8080)
	if dcm.Spec.Service.Port > 0 {
		httpPort = dcm.Spec.Service.Port
	}

	grpcPort := int32(50051)
	if dcm.Spec.Service.GRPCPort > 0 {
		grpcPort = dcm.Spec.Service.GRPCPort
	}

	// Build environment variables
	env := []corev1.EnvVar{
		{Name: "PORT", Value: fmt.Sprintf("%d", httpPort)},
		{Name: "GRPC_PORT", Value: fmt.Sprintf("%d", grpcPort)},
		{Name: "GRPC_ENABLED", Value: fmt.Sprintf("%t", grpcEnabled)},
		{Name: "THREADS", Value: fmt.Sprintf("%d", threads)},
		{Name: "LOG_FORMAT", Value: "json"},
		{Name: "MODEL_REPO", Value: dcm.Spec.Model.RepoID},
		{Name: "MODEL_FILE", Value: dcm.Spec.Model.Filename},
	}

	// Add HF token from secret if specified
	if dcm.Spec.Model.HFTokenSecretRef != nil {
		key := "HF_TOKEN"
		if dcm.Spec.Model.HFTokenSecretRef.Key != "" {
			key = dcm.Spec.Model.HFTokenSecretRef.Key
		}
		env = append(env, corev1.EnvVar{
			Name: "HF_TOKEN",
			ValueFrom: &corev1.EnvVarSource{
				SecretKeyRef: &corev1.SecretKeySelector{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: dcm.Spec.Model.HFTokenSecretRef.Name,
					},
					Key: key,
				},
			},
		})
	}

	// Add custom environment variables
	env = append(env, dcm.Spec.Env...)

	containerPorts := []corev1.ContainerPort{
		{Name: "http", ContainerPort: httpPort, Protocol: corev1.ProtocolTCP},
	}
	if grpcEnabled {
		containerPorts = append(containerPorts, corev1.ContainerPort{
			Name:          "grpc",
			ContainerPort: grpcPort,
			Protocol:      corev1.ProtocolTCP,
		})
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dcm.Name,
			Namespace: dcm.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
					Annotations: map[string]string{
						"prometheus.io/scrape": "true",
						"prometheus.io/port":   fmt.Sprintf("%d", httpPort),
						"prometheus.io/path":   "/metrics",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            "densecore",
							Image:           image,
							ImagePullPolicy: imagePullPolicy,
							Ports:           containerPorts,
							Env:             env,
							Resources:       dcm.Spec.Resources,
							LivenessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/health/live",
										Port: intstr.FromString("http"),
									},
								},
								InitialDelaySeconds: 30,
								PeriodSeconds:       10,
								TimeoutSeconds:      5,
								FailureThreshold:    3,
							},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/health/ready",
										Port: intstr.FromString("http"),
									},
								},
								InitialDelaySeconds: 10,
								PeriodSeconds:       5,
								TimeoutSeconds:      3,
								FailureThreshold:    3,
							},
							StartupProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/health/startup",
										Port: intstr.FromString("http"),
									},
								},
								InitialDelaySeconds: 5,
								PeriodSeconds:       5,
								TimeoutSeconds:      5,
								FailureThreshold:    60, // Allow 5 minutes for large model loading
							},
						},
					},
					NodeSelector:                  dcm.Spec.NodeSelector,
					Tolerations:                   dcm.Spec.Tolerations,
					Affinity:                      dcm.Spec.Affinity,
					TerminationGracePeriodSeconds: ptr(int64(60)),
				},
			},
		},
	}

	return deployment
}

// ============================================================================
// Service Reconciliation
// ============================================================================

func (r *DenseCoreModelReconciler) reconcileService(ctx context.Context, dcm *densecoreioiov1alpha1.DenseCoreModel) (*corev1.Service, error) {
	log := log.FromContext(ctx)

	service := &corev1.Service{}
	serviceName := types.NamespacedName{Name: dcm.Name, Namespace: dcm.Namespace}

	err := r.Get(ctx, serviceName, service)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}

	desired := r.buildService(dcm)
	if err := controllerutil.SetControllerReference(dcm, desired, r.Scheme); err != nil {
		return nil, err
	}

	if apierrors.IsNotFound(err) {
		log.Info("Creating Service", "name", desired.Name)
		if err := r.Create(ctx, desired); err != nil {
			return nil, err
		}
		return desired, nil
	}

	// Update if needed
	log.Info("Updating Service", "name", service.Name)
	service.Spec.Ports = desired.Spec.Ports
	service.Spec.Type = desired.Spec.Type
	if err := r.Update(ctx, service); err != nil {
		return nil, err
	}

	return service, nil
}

func (r *DenseCoreModelReconciler) buildService(dcm *densecoreioiov1alpha1.DenseCoreModel) *corev1.Service {
	labels := map[string]string{
		"app.kubernetes.io/name":       "densecore",
		"app.kubernetes.io/instance":   dcm.Name,
		"app.kubernetes.io/managed-by": "densecore-operator",
	}

	serviceType := corev1.ServiceTypeClusterIP
	if dcm.Spec.Service.Type != "" {
		serviceType = dcm.Spec.Service.Type
	}

	httpPort := int32(8080)
	if dcm.Spec.Service.Port > 0 {
		httpPort = dcm.Spec.Service.Port
	}

	grpcEnabled := true
	if dcm.Spec.Service.GRPCEnabled != nil {
		grpcEnabled = *dcm.Spec.Service.GRPCEnabled
	}

	grpcPort := int32(50051)
	if dcm.Spec.Service.GRPCPort > 0 {
		grpcPort = dcm.Spec.Service.GRPCPort
	}

	annotations := make(map[string]string)
	for k, v := range dcm.Spec.Service.Annotations {
		annotations[k] = v
	}

	ports := []corev1.ServicePort{
		{
			Name:       "http",
			Port:       httpPort,
			TargetPort: intstr.FromInt(int(httpPort)),
			Protocol:   corev1.ProtocolTCP,
		},
	}
	if grpcEnabled {
		ports = append(ports, corev1.ServicePort{
			Name:       "grpc",
			Port:       grpcPort,
			TargetPort: intstr.FromInt(int(grpcPort)),
			Protocol:   corev1.ProtocolTCP,
		})
	}

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        dcm.Name,
			Namespace:   dcm.Namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: corev1.ServiceSpec{
			Type:     serviceType,
			Selector: labels,
			Ports:    ports,
		},
	}
}

// ============================================================================
// HPA Reconciliation
// ============================================================================

func (r *DenseCoreModelReconciler) reconcileHPA(ctx context.Context, dcm *densecoreioiov1alpha1.DenseCoreModel) (*autoscalingv2.HorizontalPodAutoscaler, error) {
	log := log.FromContext(ctx)

	hpa := &autoscalingv2.HorizontalPodAutoscaler{}
	hpaName := types.NamespacedName{Name: dcm.Name, Namespace: dcm.Namespace}

	err := r.Get(ctx, hpaName, hpa)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}

	desired := r.buildHPA(dcm)
	if err := controllerutil.SetControllerReference(dcm, desired, r.Scheme); err != nil {
		return nil, err
	}

	if apierrors.IsNotFound(err) {
		log.Info("Creating HPA", "name", desired.Name)
		if err := r.Create(ctx, desired); err != nil {
			return nil, err
		}
		return desired, nil
	}

	// Update if needed
	log.Info("Updating HPA", "name", hpa.Name)
	hpa.Spec = desired.Spec
	if err := r.Update(ctx, hpa); err != nil {
		return nil, err
	}

	return hpa, nil
}

func (r *DenseCoreModelReconciler) buildHPA(dcm *densecoreioiov1alpha1.DenseCoreModel) *autoscalingv2.HorizontalPodAutoscaler {
	minReplicas := int32(1)
	if dcm.Spec.Autoscaling.MinReplicas > 0 {
		minReplicas = dcm.Spec.Autoscaling.MinReplicas
	}

	maxReplicas := int32(5)
	if dcm.Spec.Autoscaling.MaxReplicas > 0 {
		maxReplicas = dcm.Spec.Autoscaling.MaxReplicas
	}

	targetCPU := int32(70)
	if dcm.Spec.Autoscaling.TargetCPUUtilization > 0 {
		targetCPU = dcm.Spec.Autoscaling.TargetCPUUtilization
	}

	metrics := []autoscalingv2.MetricSpec{
		{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name: corev1.ResourceCPU,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: &targetCPU,
				},
			},
		},
	}

	if dcm.Spec.Autoscaling.TargetMemoryUtilization != nil {
		metrics = append(metrics, autoscalingv2.MetricSpec{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name: corev1.ResourceMemory,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: dcm.Spec.Autoscaling.TargetMemoryUtilization,
				},
			},
		})
	}

	return &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dcm.Name,
			Namespace: dcm.Namespace,
			Labels: map[string]string{
				"app.kubernetes.io/name":       "densecore",
				"app.kubernetes.io/instance":   dcm.Name,
				"app.kubernetes.io/managed-by": "densecore-operator",
			},
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       dcm.Name,
			},
			MinReplicas: &minReplicas,
			MaxReplicas: maxReplicas,
			Metrics:     metrics,
		},
	}
}

// ============================================================================
// Status Updates
// ============================================================================

func (r *DenseCoreModelReconciler) updateStatus(ctx context.Context, dcm *densecoreioiov1alpha1.DenseCoreModel, deployment *appsv1.Deployment, service *corev1.Service) {
	dcm.Status.ReadyReplicas = deployment.Status.ReadyReplicas
	dcm.Status.ObservedGeneration = dcm.Generation
	dcm.Status.LastUpdated = ptr(metav1.Now())

	// Determine phase
	if deployment.Status.ReadyReplicas > 0 {
		dcm.Status.Phase = densecoreioiov1alpha1.PhaseRunning
		dcm.Status.Message = "Model is running and ready to serve requests"

		// Set Ready condition
		meta.SetStatusCondition(&dcm.Status.Conditions, metav1.Condition{
			Type:    densecoreioiov1alpha1.ConditionTypeReady,
			Status:  metav1.ConditionTrue,
			Reason:  "Ready",
			Message: "Deployment has ready replicas",
		})
	} else if deployment.Status.Replicas > 0 {
		dcm.Status.Phase = densecoreioiov1alpha1.PhaseLoading
		dcm.Status.Message = "Model is being loaded"

		meta.SetStatusCondition(&dcm.Status.Conditions, metav1.Condition{
			Type:    densecoreioiov1alpha1.ConditionTypeProgressing,
			Status:  metav1.ConditionTrue,
			Reason:  "Loading",
			Message: "Model is being loaded into memory",
		})
	} else {
		dcm.Status.Phase = densecoreioiov1alpha1.PhasePending
		dcm.Status.Message = "Waiting for pods to be scheduled"
	}

	// Set endpoint
	if service != nil {
		port := int32(8080)
		if dcm.Spec.Service.Port > 0 {
			port = dcm.Spec.Service.Port
		}
		dcm.Status.Endpoint = fmt.Sprintf("http://%s.%s.svc:%d", service.Name, service.Namespace, port)

		grpcEnabled := true
		if dcm.Spec.Service.GRPCEnabled != nil {
			grpcEnabled = *dcm.Spec.Service.GRPCEnabled
		}
		if grpcEnabled {
			grpcPort := int32(50051)
			if dcm.Spec.Service.GRPCPort > 0 {
				grpcPort = dcm.Spec.Service.GRPCPort
			}
			dcm.Status.GRPCEndpoint = fmt.Sprintf("%s.%s.svc:%d", service.Name, service.Namespace, grpcPort)
		} else {
			dcm.Status.GRPCEndpoint = ""
		}
	}

	_ = r.Status().Update(ctx, dcm)
}

func (r *DenseCoreModelReconciler) setFailedStatus(ctx context.Context, dcm *densecoreioiov1alpha1.DenseCoreModel, reason, message string) {
	dcm.Status.Phase = densecoreioiov1alpha1.PhaseFailed
	dcm.Status.Message = message
	dcm.Status.LastUpdated = ptr(metav1.Now())

	meta.SetStatusCondition(&dcm.Status.Conditions, metav1.Condition{
		Type:    densecoreioiov1alpha1.ConditionTypeDegraded,
		Status:  metav1.ConditionTrue,
		Reason:  reason,
		Message: message,
	})

	_ = r.Status().Update(ctx, dcm)
}

// ============================================================================
// Cleanup
// ============================================================================

func (r *DenseCoreModelReconciler) cleanupResources(ctx context.Context, dcm *densecoreioiov1alpha1.DenseCoreModel) error {
	log := log.FromContext(ctx)
	log.Info("Cleaning up resources", "name", dcm.Name)

	// Resources will be garbage collected due to OwnerReference
	// This is a hook for any additional cleanup if needed
	return nil
}

// ============================================================================
// Setup
// ============================================================================

// SetupWithManager sets up the controller with the Manager.
func (r *DenseCoreModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&densecoreioiov1alpha1.DenseCoreModel{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Owns(&autoscalingv2.HorizontalPodAutoscaler{}).
		Complete(r)
}

// Helper function
func ptr[T any](v T) *T {
	return &v
}
