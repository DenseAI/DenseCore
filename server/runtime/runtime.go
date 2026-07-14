// Package runtime exposes the stable host surface used by private server
// compositions without exporting DenseCore's internal assembly packages.
package runtime

import (
	"descore-server/internal/engine"
	internalserver "descore-server/internal/server"
)

type Options = internalserver.Options
type ServerInstance = internalserver.ServerInstance

func Run(opts *Options) error                      { return internalserver.Run(opts) }
func Start(opts *Options) (*ServerInstance, error) { return internalserver.Start(opts) }
func AcquireEnterprisePlugin() error               { return engine.AcquireEnterprisePlugin() }
func ReleaseEnterprisePlugin()                     { engine.ReleaseEnterprisePlugin() }
