#!/bin/bash
# 
# Creates two environment files (one for shell and the other for CMake) that
# properly defines VCPKG_ROOT variable.
# If VCPKG_ROOT environment var (or a CLI argument --vcpkg_dir) is defined
# to a proper vcpkg root directory, this scripts does nothing else, except
# creating environment files.
# Else it installs vcpkg to a specified directory (repo_root/ext/vcpkg_$ARCH 
# is the default choice)
#
# note that though the script is invoked from CMake, it could also be used
# totally separately.

set -o errexit -o noglob -o nounset

# note: all relative dirs MUST be relative to THIS_DIR / dir of this script.
THIS_DIR="$(realpath "$(dirname "$(readlink -e "$0")")")"

# this script will create a .sh and .path files that sets up a corresponding
# VCPKG_ROOT var. This is needed to propagate VCPKG_ROOT var from this script
# further.
ENV_FILE_PFX="$THIS_DIR/vcpkg_env"

REFRESH_VCPKG=0    # even when vcpkg exists, updates it to the latest version

# need to detect OS to properly set machine architecture
echo ""
echo "Detecting OS"
if $(which lsb_release); then
    CODENAME=$(lsb_release --short --codename)
elif [ -r /etc/os-release ]; then
    . /etc/os-release
    CODENAME=$VERSION_CODENAME
elif $(which uname); then
    CODENAME=$(uname -s)
    [ "$CODENAME" = "Darwin" ] || CODENAME=$(uname --operating-system)
fi
[ -z "$CODENAME" ] && echo "Unknown or unsuported OS; aborting installation" && exit 1
echo "Found OS=$CODENAME"

if [[ "$CODENAME" == "Msys" ]]; then
    # assuming Windows always runs on x86_64
    ARCH="amd64"
    # TODO: fix for Windows here
else
    ARCH="$(dpkg --print-architecture)"
fi

if [[ -z "${VCPKG_ROOT:+x}" ]]; then
    # since the script could run in jail which could have a different architecture
    # from host, to not spoil a host vcpkg, adding an archirecture suffix to dir
    VCPKG_ROOT="$THIS_DIR/../../ext/vcpkg_$ARCH"
fi

script_name="$(basename '$0')"
# POSITIONAL_ARGS=()    # not used yet
while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--env_file_pfx) ENV_FILE_PFX="$2"; shift; shift;;
    -r|--refresh_vcpkg) REFRESH_VCPKG=1; shift;;
    -v|--vcpkg_dir) VCPKG_ROOT="$2"; shift; shift;;

    -h|--help)
        echo "Usage: $script_name [options]"
        echo "Options can be any combination of:"
        echo "-e|--env_file_pfx </abs/path> - abs path prefix to .sh and .path environment files that script"
        echo "                                will create. Default value is"
        echo "                                $ENV_FILE_PFX"
        echo "-r|--refresh_vcpkg            - do 'git pull' and 'bootstrap-vcpkg.sh' if vcpkg dir exist"
        echo "                                before proceeding"
        echo "-v|--vcpkg_dir </abs/path>    - use this abs path to vcpkg directory instead of the default"
        echo "                                $VCPKG_ROOT"
        
        exit 1;;

    -*|--*)
      echo "Unknown option $1. Add '-h' or '--help' to see options."
      exit 1;;

    *)
        echo "positional args aren't supported"
        exit 1;;
      #POSITIONAL_ARGS+=("$1") # save positional arg
      #shift;;
  esac
done
#set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
#if [[ $# -gt 0 ]] && [[ -n $1 ]]; then
    #echo "positional argument 1: $1"
#fi


setup_vcpkg(){
  echo ""
  echo "Setting up vcpkg"

  if [[ -d "$VCPKG_ROOT" ]]; then
    if (( $REFRESH_VCPKG == 0 )); then
      echo "vcpkg directory is found. Using it as is. The version is:"
      # failure of the next command will fail the script due to set -e, so it
      # also checks the dir validity that way
      "$VCPKG_ROOT/vcpkg" --version
      echo ""
    else
      echo "Refreshing it's repo..."
      echo ""
      cur_dir="$(pwd)"
      cd "$VCPKG_ROOT"
      git pull
      cd "$cur_dir"
      cur_dir=;
    fi
  else
    echo "Cloning it's repo..."
    echo ""    
    git clone https://github.com/Microsoft/vcpkg.git "$VCPKG_ROOT"
    # note that latest versions of vcpkg is/was buggy on Windows and just crash
    # I've made a bug report here https://github.com/microsoft/vcpkg/issues/30046
    # so it'd be nice if someone ping them also.
    # If vcpkg crashes, one might need to checkout some older version with, f.e.
    # git checkout e809a42f87565e803b2178a0c11263f462d1800a
    REFRESH_VCPKG=1
  fi

  VCPKG_ROOT=$(realpath $VCPKG_ROOT)

  if (( $REFRESH_VCPKG != 0 )); then
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
  fi

  # saving VCPKG_ROOT value to environment files
  cat > "$ENV_FILE_PFX.sh" << EOF
#!/bin/bash
export VCPKG_ROOT="$VCPKG_ROOT"
EOF
  chmod 755 "$ENV_FILE_PFX.sh"

  echo "$VCPKG_ROOT" > "$ENV_FILE_PFX.path"
}

setup_vcpkg

echo ""
echo "Congratulations, you should be able to use vcpkg now."
echo "To setup the shell environment, execute this command in your current terminal:"
echo ""
echo "$ source '$ENV_FILE_PFX.sh'"
echo ""
echo "Alternatively for CMake, read path to vcpkg from $ENV_FILE_PFX.path"
echo ""

