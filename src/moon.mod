name = "riantr/ordinarydiffeq"

version = "0.1.0"

license = "Apache-2.0"

keywords = [
  "ode",
  "scientific-computing",
  "numerical",
  "wasm",
  "js",
  "native",
]

preferred_target = "native"

supported_targets = [ "js", "wasm-gc", "native" ]

source = "src"

description = "A MoonBit port of OrdinaryDiffEq.jl: adaptive ODE solvers (Tsit5, Verner, Rosenbrock, BDF) with dense output, step-size control and events."
