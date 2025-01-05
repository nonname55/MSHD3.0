<template>
  <div class="tweet-prediction-container">
    <el-form ref="tweetForm" :model="tweetData" label-width="100px">
      <el-form-item label="关键词">
        <el-input v-model="tweetData.keyword" placeholder="输入关键词 (可选)"></el-input>
      </el-form-item>
      
      <el-form-item label="位置">
        <el-input v-model="tweetData.location" placeholder="输入位置信息 (可选)"></el-input>
      </el-form-item>
      
      <el-form-item label="推文内容" required>
        <el-input
          v-model="tweetData.text"
          type="textarea"
          :rows="4"
          placeholder="请输入推文内容"
        ></el-input>
      </el-form-item>

      <el-form-item>
        <el-button 
          type="primary" 
          @click="submitPrediction"
          :disabled="!tweetData.text"
        >
          进行预测
        </el-button>
      </el-form-item>
    </el-form>

    <!-- 预测结果展示 -->
    <div v-if="predictionResult" class="result-container">
      <el-card class="result-card">
        <template #header>
          <div class="card-header">
            <span class="title">预测结果</span>
          </div>
        </template>
        <div class="result-content">
          <div class="prediction-details">
            <p class="prediction-text">
              <!-- 使用标签显示灾情与否 -->
              <span :class="['tag-custom', predictionResult.is_disaster ? 'danger' : 'success']">
                {{ predictionResult.is_disaster ? '真实灾情' : '非灾情' }}
              </span>
            </p>
            <p class="prediction-text">
              置信度: {{ predictionResult.confidence }}
            </p>
            <p v-if="predictionResult.location" class="prediction-text">
              位置信息: {{ predictionResult.location }}
            </p>
            <p v-if="predictionResult.message" class="warning-message">
              {{ predictionResult.message }}
            </p>
          </div>
          <div class="timestamp">预测时间: {{ predictionTime }}</div>
        </div>
      </el-card>
    </div>
  </div>
</template>


<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

interface TweetData {
  keyword: string
  location: string
  text: string
}

interface PredictionResult {
  is_disaster: boolean
  prediction: string
  confidence: string
  location: string | null
  text: string
  message?: string
}

const tweetData = reactive<TweetData>({
  keyword: '',
  location: '',
  text: ''
})

const predictionResult = ref<PredictionResult | null>(null)
const predictionTime = ref<string>('')

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

const submitPrediction = async () => {
  try {
    if (!tweetData.text.trim()) {
      ElMessage.warning('请输入推文内容')
      return
    }

    // 发送POST请求到后端进行推文分类
    const response = await axios.post('http://localhost:5000/classify/tweet', {
      text: tweetData.text,
      location: tweetData.location,
      keyword: tweetData.keyword
    })

    console.log('后端返回的内容:', response.data)
    
    if (response.data.success) {
      // 根据 target 字段判断是灾情还是非灾情
      const target = response.data.data.target
      const isDisaster = target === 1
      predictionResult.value = {
        is_disaster: isDisaster,
        prediction: isDisaster ? '真实灾情' : '非灾情',
        confidence: response.data.data.confidence,
        location: target === 1 ? response.data.data.location : null,
        text: response.data.data.text,
        message: isDisaster && response.data.data.location ? `检测到真实灾害！位置信息：${response.data.data.location}` : undefined
      }
      predictionTime.value = formatDateTime()
    } else {
      ElMessage.error('预测失败: ' + response.data.error)
    }

  } catch (error: any) {
    console.error('预测失败:', error)
    ElMessage.error(`预测失败: ${error.response?.data?.error || error.message}`)
  }
}
</script>

<style scoped>
.tweet-prediction-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.result-container {
  margin-top: 30px;
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
  padding: 15px 0;
}

.prediction-details {
  margin: 15px 0;
}

.prediction-text {
  font-size: 16px;
  color: #409EFF;
  margin-bottom: 12px; /* 增加与其他元素的间距 */
  font-weight: 500;
}

.warning-message {
  color: #F56C6C;
  font-size: 14px;
  margin-top: 10px;
  padding: 8px;
  background-color: #FEF0F0;
  border-radius: 4px;
}

.timestamp {
  font-size: 12px;
  color: #909399;
  text-align: right;
  margin-top: 10px;
}

/* 新增的美化部分 */
.tag-custom {
  font-weight: bold;
  font-size: 16px;
  padding: 8px 16px;
  border-radius: 20px;
  margin-bottom: 12px; /* 添加间距，避免与置信度栏太近 */
}

.tag-custom.danger {
  background-color: #f56c6c;
  color: #fff;
}

.tag-custom.success {
  background-color: #67C23A;
  color: #fff;
}
</style>

