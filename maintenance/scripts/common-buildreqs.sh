#!/bin/bash
#
# This script ensures the most basic stuff necessary for building is installed

set -o errexit -o noglob -o nounset

# Note: all relative dirs MUST be relative to THIS_DIR / dir of this script.
THIS_DIR="$(realpath "$(dirname "$(readlink -e "$0")")")"

SUDO=sudo
NON_INTERACTIVE=--yes


# pip contains the latest version which is (almost) always good, but it may
# trigger too much paranoia
CPPLINT_FROM_PIP=1
# fixing version to reduce paranoia a bit
CPPLINT_VERSION_PIP="1.6.1"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c)    CROSS_COMPILE=yes; shift;;
        --cpplint_apt)  CPPLINT_FROM_PIP=0; shift;;
        --cpplint_ver)  CPPLINT_VERSION_PIP="$2" CPPLINT_FROM_PIP=1; shift; shift;;
        -i)    NON_INTERACTIVE=''; shift;;
        -n)    SUDO=''; shift;;

        *)     exit 2;;
    esac
done

#
# Distro'es build dependencies
#
do_focal() {
    DISTR="$1"

    if (( $CPPLINT_FROM_PIP == 1 )); then
        ADDITIONAL_PKGS=""
    else
        ADDITIONAL_PKGS="cpplint"   
    fi

    $SUDO apt update

    $SUDO apt install NON_INTERACTIVE \
        build-essential \
        clang-format-12 \
        clang-tidy-12 \
        cmake \
        cppcheck \
        curl \
        git \
        ninja-build \
        tar \
        unzip \
        wget \
        zip \
        $ADDITIONAL_PKGS

    if (( $CPPLINT_FROM_PIP == 1 )); then
        python3 -m pip install "cpplint==$CPPLINT_VERSION_PIP"
    fi
}

do_jammy() {
    do_focal "$1"
}

do_Msys() {
    echo "nothing to do here yet"
}

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

echo "Installing stuff for '$CODENAME'..."
do_${CODENAME} "${CODENAME}"
echo ""

"$THIS_DIR/setup_vcpkg.sh"
