#!/bin/bash
#
# This script deletes everything except `vcpkg_installed` directory in the dir passed as $1 arg.
# This is useful to do instead of "rm -rf ./build" since saving `vcpkg_installed` saves a lot of
# runtime for vcpkg.
# Pass --silent or -s as $2 to confirm directory deletion in unattended mode.

set -o errexit -o noglob -o nounset

[[ "$#" -lt 1  ]] && { echo "Pass argument 1 as a directory to clean"; exit 1; }

# if the dir doesn't exist - that's ok, return success code, since the job is done anyway
if [[ ! -d "$1" ]]; then
    echo "Dir from argument 1 doesn't exist, that's fine, just exiting."
    exit 0
fi

# the check above is a must for realpath...
BUILD_DIR="$(realpath $1)"
# now the final test after realpath
[[ -z "${BUILD_DIR:+x}" || ! -d "$BUILD_DIR" ]] && { echo "Argument 1 must be a directory to clean. Ignoring and doing nothing."; exit 0; }

if [[ "$#" -lt 2 || ("$2" != "--silent" && "$2" != "-s") ]]; then
    read -p "Clean build directory '${BUILD_DIR}'? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || { echo "Did nothing"; exit 1; }
    echo "Proceeding..."
fi

BUILD_VCPKG_INSTALLED="$BUILD_DIR/vcpkg_installed"

TEMP_DIR=""
if [[ -d "$BUILD_VCPKG_INSTALLED" ]]; then
    # TEMP_DIR=$(mktemp -ud)
    # ^^ works, but the tmp dir could be spawned on a different disk which will make the process slooow
    TEMP_DIR="$(realpath $BUILD_DIR/..)/vcpkg_tmp_cache"

    echo "Saving vcpkg cache from $BUILD_VCPKG_INSTALLED"
    echo "to $TEMP_DIR"
    [[ -d "$TEMP_DIR" ]] && { echo "Temporary dir $TEMP_DIR exist. Remove it first!"; exit 2; }
    mv "$BUILD_VCPKG_INSTALLED" "$TEMP_DIR"
fi

rm -rf "$BUILD_DIR"

mkdir -p "$BUILD_DIR"

if [[ "$TEMP_DIR" != "" ]]; then
    echo "Restoring vcpkg cache dir"
    mv "$TEMP_DIR" "$BUILD_VCPKG_INSTALLED" || {
        # sometimes ^^ fails in WSL with "Permission denied". Can't find why is that happening.
        echo "Sometimes it ^^ fails. Sleeping a bit usually helps..."
        sleep 2
        mv "$TEMP_DIR" "$BUILD_VCPKG_INSTALLED"
        echo "It worked, just ignore the error ^^"
    }
fi

echo "Done!"
