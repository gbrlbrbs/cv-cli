{
  description = "A flake providing a dev shell for CUDA and CUDA development using NVCC.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Adjust if needed
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          bashInteractive
          cudatoolkit
          cudaPackages.cudnn
          cudaPackages.cuda_cudart

          # Need to explicitly override the system gcc (gcc14 in this case) as CUDA requires a
          # lower version for compatibility
          gcc13
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudatoolkit}

          # Set CC to GCC 13 to avoid the version mismatch error
          export CC=${pkgs.gcc13}/bin/gcc
          export CXX=${pkgs.gcc13}/bin/g++
          export PATH=${pkgs.gcc13}/bin:$PATH

          # Add necessary paths for dynamic linking
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              "/run/opengl-driver" # Needed to find libGL.so
              pkgs.cudatoolkit
              pkgs.cudaPackages.cudnn
            ]
          }:$LD_LIBRARY_PATH

          # Set LIBRARY_PATH to help the linker find the CUDA static libraries
          export LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              pkgs.cudatoolkit
            ]
          }:$LIBRARY_PATH
        '';
      };
    };
}
