# Tinyjax

JAX + Tinygrad = Run everywhere

This is a prototype tinygrad backend for JAX. Tinygrad is a simple ML framework
and compiler that supports many devices such as CUDA, OpenCL, Metal, and even
WebGPU and C. JAX is a powerful ML framework that dispatches operations to XLA.
By running those operations using Tinygrad instead, we enjoy the extended device
support from Tinygrad:

- Run JAX on OpenCL (e.g. Intel GPU, AMD GPU without any experimental JAX builds)
- Run JAX on Apple Silicon/Metal without experimental prebuilts
- Compile JAX to WebGL and WebGPU with fused kernels
- Compile JAX to C

## How does it work?

The [Tinygrad] API builds a lazy computation graph by tracking what APIs are
called on the tensors. When somebody actually needs the data, the graph is JIT
compiled into one or more kernels and scheduled on the device (called "realize").
There are only 25 fundamental operations that everything else lowers into, which
makes it very easy to add new backends.

JAX can turn a Python function into Jaxpr by tracing it with abstract values,
similar to Tinygrad. The resulting Jaxpr is a functional expression language that
is strongly related to XLA.

We implement a Jaxpr interpreter that translates each Jaxpr operation (such as
`dot_general`) into a Tinygrad operation (e.g. `einsum`). Because Tinygrad
operations are lazy, the output of the interpreter is a computation graph instead
of concrete values. And the graph can be JIT compiled into one big GPU kernel via
Tinygrad codegen.

## Current state

Right now enough ops are implemented to convert a ConvNet (see `ops.py`). But it
is very straightforward to add new ops.

## Examples

See [README.ipynb][README.ipynb].

[Tinygrad]: https://github.com/tinygrad/tinygrad
[README.ipynb]: ./README.ipynb
