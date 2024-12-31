<script setup lang="ts">
defineOptions({
  name: "Welcome"
});
import { ref, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'

const chartRef = ref<HTMLElement>()
let myChart: echarts.ECharts | null = null

const option: EChartsOption = {
  title: {
    text: 'ECharts 入门示例'
  },
  tooltip: {},
  xAxis: {
    data: ['衬衫', '羊毛衫', '雪纺衫', '裤子', '高跟鞋', '袜子']
  },
  yAxis: {},
  series: [
    {
      name: '销量',
      type: 'bar',
      data: [5, 20, 36, 10, 10, 20]
    }
  ]
}

onMounted(() => {
  if (chartRef.value) {
    myChart = echarts.init(chartRef.value)
    myChart.setOption(option)
  }
})

// 防止内存泄漏
onBeforeUnmount(() => {
  if (myChart) {
    myChart.dispose()
    myChart = null
  }
})
</script>

<template>
  <h1>Gaia-Home Page</h1>
  <div ref="chartRef" style="width: 600px; height: 400px;"></div>
</template>
