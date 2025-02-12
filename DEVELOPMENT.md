### Requirements

Install [nix](https://nixos.org/download.html#nix-install-linux) to setup adhoc
development environment.

Start new shell:

```bash
nix-shell
```

We are using [bazel](https://bazel.build/) as a build system. I also provided a `tools/bazel` script to make it easier to use.

Build the extension:

```bash
./tools/bazel run //:extensions
```

### Testing

Run tests:

```bash
./tools/bazel test //tests/...
```

> NOTE: Make sure to include the `extern/whispercpp`, `extern/pybind11/include`, and ```$(python3-config --prefix)/include/python3```  in your `CPLUS_INCLUDE_PATH` so that `clangd` can find the headers in your editor.
