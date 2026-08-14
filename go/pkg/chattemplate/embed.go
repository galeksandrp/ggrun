package chattemplate

import (
	"embed"
)

//go:embed templates/*.jinja
var templatesFS embed.FS
