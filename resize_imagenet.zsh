
PARENT_DIR="BABY_IMAGENET"
IMAGE_SIZE=300
OUTPUT_DIR="baby_imagenet_${IMAGE_SIZE}"

# Set to any value to have a dry run
DRY_RUN=

for i in  ${PARENT_DIR}/*/* ; do
    if [[ ! $DRY_RUN ]] ; then
        mkdir -p "${i/${PARENT_DIR}/${OUTPUT_DIR}}" 
    else
        echo "${i/${PARENT_DIR}/${OUTPUT_DIR}}"
    fi
done

for i in  ${PARENT_DIR}/*/*/*.jpg ; do
    if [[ ! $DRY_RUN ]] ; then
        if [[ ! -e "${i/${PARENT_DIR}/${OUTPUT_DIR}}" ]]; then
            echo "${i/${PARENT_DIR}/${OUTPUT_DIR}}"
            convert -verbose "${i}" -resize "${IMAGE_SIZE}x${IMAGE_SIZE}>" "${i/${PARENT_DIR}/${OUTPUT_DIR}}" && echo "DONE!!"
        else
            echo "Skipping" 
        fi
    else
        echo "${i}" "${IMAGE_SIZE}x${IMAGE_SIZE}>" "${i/${PARENT_DIR}/${OUTPUT_DIR}}"
    fi
done


echo "Testing file integrity ..."
for i in  ${OUTPUT_DIR}/*/*/*.jpg ; do
    identify "${i}" > /dev/null || echo "${i} is broken"
done


echo "Renaming to checksum"
for i in  ${OUTPUT_DIR}/*/*/*.jpg ; do
    mv --verbose ${i} "$(dirname ${i})/$(md5sum ${i} | sed -e "s/\s.*//g").jpg"

done


du -sh ${OUTPUT_DIR}/*

echo "Number of Images"
echo "Train"
ls ${OUTPUT_DIR}/train/*/*.jpg | wc -l
echo "Test"
ls ${OUTPUT_DIR}/test/*/*.jpg | wc -l