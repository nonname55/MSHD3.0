<template>
  <el-upload
    ref="upload"
    class="upload-demo"
    action="http://localhost:5000/classify"
    :limit="1"
    :on-exceed="handleExceed"
    :auto-upload="false"
    :on-success="handleSuccess"
    :on-change="handleChange"
    :file-list="fileList"
    list-type="picture"
  >
    <template #trigger>
      <el-button type="primary">select file</el-button>
    </template>
    <el-button class="ml-3" type="success" native-type="button" @click="submitUpload($event)">
      upload to server
    </el-button>
    <template #tip>
      <div class="el-upload__tip text-red">
        limit 1 file, new file will cover the old file
      </div>
    </template>
  </el-upload>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { genFileId, ElMessage } from 'element-plus'  // 正确导入 ElMessage
import type { UploadInstance, UploadProps, UploadRawFile, UploadUserFile } from 'element-plus'

const upload = ref<UploadInstance>()
const fileList = ref<UploadUserFile[]>([])

const handleExceed: UploadProps['onExceed'] = (files) => {
  upload.value!.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  upload.value!.handleStart(file)
}

const handleSuccess: UploadProps['onSuccess'] = (
  response,
  uploadFile,
  uploadFiles
) => {
  if (uploadFile.raw) {
    uploadFile.url = URL.createObjectURL(uploadFile.raw)
  }
  fileList.value = [...uploadFiles]
}

const handleChange: UploadProps['onChange'] = (uploadFile, uploadFiles) => {
  if (uploadFile.raw) {
    uploadFile.url = URL.createObjectURL(uploadFile.raw)
  }
  fileList.value = [...uploadFiles]
}

const submitUpload = async (event) => {
  if (event) {
    event.preventDefault(); // 阻止默认行为
  }

  // 如果没有选择文件，给出提示
  if (fileList.value.length === 0) {
    ElMessage.warning('请先选择文件');
    return;
  }

  // 创建 FormData
  const formData = new FormData();
  const file = fileList.value[0].raw; // 获取原始文件
  if (file) {
    formData.append('file', file);
  }

  try {
    // 发起上传请求
    const response = await fetch('http://localhost:5000/classify', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 解析返回的 JSON 数据
    const result = await response.json();

    // 打印后端返回的内容
    console.log('后端返回的内容:', result);

    // 根据返回内容给用户提示
    ElMessage.success({
      message: `分类结果: ${result.msg || result}`,
      duration: 5000  // 设置持续时间为 5 秒（5000 毫秒）
    });
  } catch (error) {
    console.error('上传失败:', error);
    ElMessage.error(`上传失败: ${error.message}`);
  }
};


</script>