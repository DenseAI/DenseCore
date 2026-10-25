#pragma once
#include <string>
struct EngineState;
// Called before returning a newly initialized engine to callers. Selection is
// published atomically and remains fixed throughout that engine's lifetime.
bool ConfigureSpeculation(EngineState* state, int mode, int draft_n_max, std::string* error);
