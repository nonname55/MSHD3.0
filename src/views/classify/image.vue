<template>
  <div class="upload-container">
    <el-upload
      ref="upload"
      class="upload-demo"
      action="#"
      :auto-upload="false"
      :limit="1"
      :on-exceed="handleExceed"
      :on-change="handleChange"
      :file-list="fileList"
      :http-request="customUpload"
      list-type="picture"
      :before-upload="beforeUpload"
    >
      <template #trigger>
        <el-button type="primary">选择文件</el-button>
      </template>
    </el-upload>
   
    <el-button
      class="ml-3"
      type="success"
      @click.prevent="submitUpload"
      :disabled="fileList.length === 0"
    >
      上传到服务器
    </el-button>

    <div class="el-upload__tip text-red mt-2">
      限制1个文件
    </div>

    <!-- Result Display Section -->
    <div v-if="predictionResult" class="result-container">
      <el-card class="result-card">
        <template #header>
          <div class="card-header">
            <span class="title">预测结果</span>
            <el-tag 
              :type="predictionResult ? 'success' : 'info'"
              class="ml-2"
            >
              {{ fileList[0]?.name || '未知文件' }}
            </el-tag>
          </div>
        </template>
        <div class="result-content">
          <p class="prediction-text">{{ predictionResult }}</p>
          <p class="prediction-text">{{ predictionConfidence }}</p>
          <p class="prediction-text">{{ predictionFileName }}</p>
          <div class="timestamp">预测时间: {{ predictionTime }}</div>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { genFileId, ElMessage } from 'element-plus'
import type { UploadInstance, UploadProps, UploadRawFile, UploadUserFile } from 'element-plus'
import axios from 'axios'

const upload = ref<UploadInstance>()
const fileList = ref<UploadUserFile[]>([])
const predictionResult = ref<string>('')
const predictionConfidence = ref<string>('')
const predictionFileName = ref<string>('')
const predictionTime = ref<string>('')

const beforeUpload = (file: UploadRawFile) => {
  return false
}

const customUpload = async (options: any) => {
  return false
}

const handleExceed: UploadProps['onExceed'] = (files) => {
  upload.value!.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  upload.value!.handleStart(file)
}

const handleChange: UploadProps['onChange'] = (uploadFile, uploadFiles) => {
  if (uploadFile.raw) {
    uploadFile.url = URL.createObjectURL(uploadFile.raw)
  }
  fileList.value = [...uploadFiles]
  predictionResult.value = ''
  predictionConfidence.value = ''
  predictionFileName.value = ''
  predictionTime.value = ''
}

const formatDateTime = () => {
  const now = new Date()
  return now.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

// 方法1：使用 axios
const submitUpload = async () => {
  try {
    if (fileList.value.length === 0) {
      ElMessage.warning('请先选择文件')
      return
    }

    const formData = new FormData()
    const file = fileList.value[0].raw
    if (!file) {
      ElMessage.error('文件获取失败')
      return
    }

    formData.append('file', file)
    
    const response = await axios.post('http://localhost:5000/classify/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    console.log('后端返回的内容:', response.data)
    
    const { predicted_class, confidence, filename } = response.data.data
    predictionResult.value = `预测标签: ${predicted_class}`
    predictionConfidence.value = `置信度: ${confidence}`
    predictionFileName.value = `文件名: ${filename}`
    predictionTime.value = formatDateTime()

  } catch (error: any) {
    console.error('上传失败:', error)
    ElMessage.error(`上传失败: ${error.response?.data?.message || error.message}`)
  }
}
</script>

<style scoped>
.upload-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-width: 500px;
}

.el-upload-list {
  margin: 10px 0;
}

.result-container {
  margin-top: 20px;
}

.result-card {
  background-color: #f8f9fa;
  border-radius: 8px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.title {
  font-size: 16px;
  font-weight: bold;
  color: #303133;
}

.result-content {
  padding: 10px 0;
}

.prediction-text {
  font-size: 18px;
  color: #409EFF;
  margin-bottom: 10px;
  font-weight: 500;
}

.timestamp {
  font-size: 12px;
  color: #909399;
  text-align: right;
}
</style>
