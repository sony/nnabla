BINARY_PATH=$(find . -name "test_nbla_utils")
LD_PATH=$(find . -name "libnnabla_util*.*")

test_nbla_utils=''

for b_p in ${BINARY_PATH}; do
    test_nbla_utils="$(dirname $b_p)/test_nbla_utils"
    export PATH=$(dirname $b_p):$PATH
    break
done

for ld_p in ${LD_PATH}; do
    export LD_LIBRARY_PATH=$(dirname $ld_p):$LD_LIBRARY_PATH
    break
done

echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

chmod u+x $test_nbla_utils
$test_nbla_utils
