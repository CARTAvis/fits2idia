
# The following function loads a module and save its name in an environment variable that then will
# be used to add those module loads in the modulefile itself.
function module_load {
    BU_LOADED_MODULES="$BU_LOADED_MODULES $@"
    module load $@
}

function print_run {
	echo "Executing $@"
	$@
}

function get_commit_hash {
    COMMIT_HASH=""
    if [ -d .git ] && !  [ "`which git`" = "git not found" ]; then
        COMMIT_HASH=`git log -n 1 --pretty=oneline | cut -d" " -f1`
    fi
}

# This function reads the input arguments of a build script and use them to
# define the INSTALL_DIR and MODULEFILE_DIR environment variables
function create_modulefile {
    if ! [ -n "$MODULEFILE_DIR" ]; then
        echo "MODULEFILE_DIR variable not set."
        exit 1 
    fi
    if ! [ -n "$INSTALL_DIR" ]; then
        echo "INSTALL_DIR variable not set."
        exit 1 
    fi

    if ! [ -n "$PROGRAM_NAME" ]; then
        echo "PROGRAM_NAME variable not set."
        exit 1 
    fi
    if ! [ -n "$PROGRAM_VERSION" ]; then
        PROGRAM_VERSION=devel
    fi 
    MODULE_LOAD_COMMANDS=""
    for module in $BU_LOADED_MODULES;
    do
    MODULE_LOAD_COMMANDS="$MODULE_LOAD_COMMANDS load('$module');"
    done
    echo "-- Modulefile for $PROGRAM_NAME.
local root_dir = '$INSTALL_DIR'
if (mode() ~= 'whatis') then
    $MODULE_LOAD_COMMANDS
    prepend_path('PATH', root_dir .. '/bin')
    prepend_path('LD_LIBRARY_PATH', root_dir .. '/lib')
    prepend_path('LD_LIBRARY_PATH', root_dir .. '/lib64')
    prepend_path('LIBRARY_PATH', root_dir .. '/lib')
    prepend_path('LIBRARY_PATH', root_dir .. '/lib64')
    prepend_path('CPATH', root_dir .. '/include')
    $ADDITIONAL_MODULEFILE_COMMANDS
end
    " > "$MODULEFILE_DIR/${PROGRAM_VERSION}.lua"
}

#
# process_build_script_input
#
# Description
# -----------
# Implements the build  

function process_build_script_input {
    if ! [ -n "$PROGRAM_NAME" ]; then
        echo "PROGRAM_NAME variable not set."
        exit 1 
    fi   
    if ! [ -n "$PROGRAM_VERSION" ]; then
        echo "PROGRAM_VERSION variable not set."
        exit 1
    fi

    if [ $PAWSEY_CLUSTER = "setonix" ]; then
        if [ "$1" = 'group' ]; then
            INSTALL_DIR="/software/projects/$PAWSEY_PROJECT/setonix/2025.08/development/$PROGRAM_NAME/$PROGRAM_VERSION"
            MODULEFILE_DIR="/software/projects/$PAWSEY_PROJECT/setonix/2025.08/modules/zen3/gcc/14.2.0/$PROGRAM_NAME"
        elif [ "$1" = 'user' ]; then
            INSTALL_DIR="/software/projects/$PAWSEY_PROJECT/$USER/setonix/2025.08/development/$PROGRAM_NAME/$PROGRAM_VERSION"
            MODULEFILE_DIR="/software/projects/$PAWSEY_PROJECT/$USER/setonix/2025.08/modules/zen3/gcc/14.2.0/$PROGRAM_NAME"
        elif [ "$1" != 'test' ]; then
            echo "Error parsing build script input: first parameter not recognised."
            exit 1
        fi
    else
        echo "Error: cluster not recognised."
        exit 1
    fi
    # irrespective of the cluster, if i is a test build, then the installation path is relative to the build one.
    if [ "$1" = 'test' ]; then
        INSTALL_DIR="`pwd`/build"
        MODULEFILE_DIR="$INSTALL_DIR/modulefiles/$PROGRAM_NAME"
        
        if [[ -n "$2" && "$2" != "-" ]]; then
            INSTALL_DIR="$2"
            MODULEFILE_DIR="$INSTALL_DIR/modulefiles/$PROGRAM_NAME"
        fi
        if [[ -n "$3" && "$3" != "-" ]]; then
            MODULEFILE_DIR=$3
        fi
    fi
    
    # create directories if they don't exist
    [ -d $INSTALL_DIR ] || mkdir -p $INSTALL_DIR
    [ -d $MODULEFILE_DIR ] || mkdir -p $MODULEFILE_DIR
}

# export MODULEFILE_DIR="./"
# export INSTALL_DIR="./"
# PROGRAM_NAME=fits2idia

# echo "Create the modulefile in $MODULEFILE_DIR (or $INSTALL_DIR)"
# export ADDITIONAL_MODULEFILE_COMMANDS="prepend_path('BLINK_IMAGER_PATH', root_dir )"
# create_modulefile

