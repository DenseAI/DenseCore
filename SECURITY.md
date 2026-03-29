# Security Policy

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in DenseCore, please report it responsibly.

### ⚠️ Important: Do NOT Open Public Issues

**Please do NOT open public issues for security vulnerabilities.** Public disclosure before a fix is available puts all users at risk.

### How to Report

Send your report via email to: **security@densecore.dev** (or **jwsong9294@gmail.com**)

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

| Timeline | Action |
|----------|--------|
| **24 hours** | Acknowledge receipt of your report |
| **72 hours** | Initial assessment and severity classification |
| **7 days** | Detailed response with remediation plan |
| **90 days** | Public disclosure (coordinated with you) |

For **critical vulnerabilities**, we aim to release a patch within **48 hours** of confirmation.

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
- ✅ Official Docker images (`denseai/densecore`)
- ✅ Helm charts (`charts/densecore`)
- ✅ GitHub Actions workflows

### Out of Scope

- ❌ Vulnerabilities in upstream dependencies (report to upstream, notify us)
- ❌ Social engineering attacks
- ❌ Physical security attacks
- ❌ Issues in experimental/unreleased features (marked as `[EXPERIMENTAL]`)
- ❌ Self-XSS or issues requiring physical access

## Security Best Practices

When deploying DenseCore in production:

### Authentication & Authorization
```bash
# Always enable authentication
AUTH_ENABLED=true
API_KEY_HASH=sha256:your-hashed-key  # Never use plaintext keys
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

## Bug Bounty Program

> [!NOTE]
> We are working on establishing a formal bug bounty program. In the meantime, we recognize and credit security researchers who responsibly disclose vulnerabilities.

### Recognition

Security researchers who report valid vulnerabilities will be:
- Credited in our release notes (unless anonymity is requested)
- Listed in our [SECURITY_ACKNOWLEDGMENTS.md](SECURITY_ACKNOWLEDGMENTS.md) hall of fame
- Considered for invitation to our private security advisory list

## Security Updates

Subscribe to security announcements:
- **GitHub Security Advisories**: Watch this repository
- **Mailing List**: security-announce@densecore.dev (coming soon)

---

**Thank you for helping keep DenseCore secure! 🔐**
