# Security and privacy

Do not submit patient information, clinical text, credentials, model weights,
or private infrastructure details in a public issue.

The example web client must receive `DDHIGAN_API_URL` and `DDHIGAN_API_KEY`
only from server-side secret storage. The API intentionally logs request
metadata only and its public application routes require an API key.

If you discover a security or privacy issue, use GitHub's private vulnerability
reporting feature for this repository instead of opening a public issue.
