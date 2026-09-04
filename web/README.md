# DDHIgAN Shiny client

This is the source used for the identifier-free public DDHIgAN interface on
Posit Connect Cloud. It contains a 1,640-byte aggregate-only descriptive LMM
artifact for plotting; that artifact is separate from risk inference and
contains no patient rows or identifiers.

Required server-side Secret Variables:

- `DDHIGAN_API_URL`: authenticated HTTPS inference API base URL.
- `DDHIGAN_API_KEY`: dedicated high-entropy API key.

Optional variables:

- `DDHIGAN_CA_BUNDLE`: CA bundle for a privately trusted endpoint.
- `DDHIGAN_WEB_ENV_FILE`: path to an environment file for a controlled local
  installation. It is unset by default and should not be used on Connect Cloud.
- `DDHIGAN_HISTORY_LMM_PATH`: override for the bundled aggregate LMM artifact.

Never commit variable values, an environment file, patient records, clinical
text, predictions, or a model checkpoint. Generate `manifest.json` with
`rsconnect::writeManifest()` after dependency changes.
