# Security Policy

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in DenseCore, please report it responsibly.

### ⚠️ Important: Do NOT Open Public Issues

**Please do NOT open public issues for security vulnerabilities.** Public disclosure before a fix is available puts all users at risk.

### How to Report

Send reports to [jwsong9294@gmail.com](mailto:jwsong9294@gmail.com). Do not include
an exploit or sensitive details in a public issue. If GitHub private security
advisories are enabled later, that channel may also be used.

Use the following format:

```
Subject: [SECURITY] Brief description

## Vulnerability Summary
[Concise description of the vulnerability]

## Affected Components
- [ ] C++ Core (`core/`)
- [ ] Python SDK (`python/`)
- [ ] Go Server (`server/`)
- [ ] Docker Images
- [ ] Kubernetes/Helm Charts

## Severity Assessment
[Your assessment: Critical / High / Medium / Low]

## Steps to Reproduce
1. 
2. 
3. 

## Potential Impact
[What could an attacker do with this vulnerability?]

## Suggested Fix (Optional)
[If you have ideas for how to fix this]
```

### What We Commit To

We will acknowledge private reports, assess severity, and coordinate remediation and disclosure on a case-by-case basis. Response time depends on the issue and the amount of follow-up required.

### Severity Classification

| Severity | Description | Examples |
|----------|-------------|----------|
| **Critical** | Remote code execution, data breach | Buffer overflow in inference, auth bypass |
| **High** | Significant impact, exploitation possible | Memory leak leading to DoS, info disclosure |
| **Medium** | Limited impact, requires specific conditions | SSRF with limited scope, timing attacks |
| **Low** | Minimal impact, difficult to exploit | Minor info leak, configuration issues |

## Scope

This security policy applies to:

- ✅ `core/` - C++ inference engine
- ✅ `python/` - Python SDK
- ✅ `server/` - Go REST server
- ✅ Container images built from this repository
- ✅ Helm charts (`charts/densecore`)
- ✅ GitHub Actions workflows

### Out of Scope

- ❌ Vulnerabilities in upstream dependencies (report to upstream, notify us)
- ❌ Social engineering attacks
- ❌ Physical security attacks
- ❌ Issues in experimental/unreleased features that are explicitly marked unsupported
- ❌ Self-XSS or issues requiring physical access

## Security Best Practices

When deploying DenseCore in production:

### Authentication & Authorization
```bash
# Always enable authentication
AUTH_ENABLED=true
API_KEYS=sk-example:user:default  # key:user:tier; inject the real value from a secret store
```

### Network Security
- **Use HTTPS**: Place behind a reverse proxy (nginx, Traefik) with TLS 1.3
- **Network isolation**: Run in private networks, expose only through ingress
- **Rate limiting**: Enable built-in rate limiting or use Redis-backed limits

### Container Security
```yaml
# Kubernetes: Run as non-root with read-only filesystem
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  capabilities:
    drop: ["ALL"]
```

### Resource Limits
```yaml
# Set appropriate limits to prevent DoS
resources:
  limits:
    cpu: "4"
    memory: "8Gi"
  requests:
    cpu: "2"
    memory: "4Gi"
```

### Logging & Monitoring
- Enable structured logging for audit trails
- Monitor for unusual patterns (high error rates, slow responses)
- Set up alerts for security-relevant events

## Security Updates

Follow the repository's security advisory workflow if it is enabled.

---

**Thank you for helping keep DenseCore secure.**
