<template>
  <div class="echarts-box">
    <div id="chart1" :style="{ width: this.width, height: this.height }"></div>
    <div id="chart2" :style="{ width: this.width, height: this.height }"></div>
  </div>
</template>

<script>
import * as echarts from "echarts";
import {onMounted, onUnmounted} from "vue";

export default {
  name: "App",
  props: ["width", "height"],
  setup() {
    let base1 = +new Date(1988, 9, 3);
    let oneDay1 = 24 * 3600 * 1000;
    let data1 = [[base1, Math.random() * 300]];
    for (let i = 1; i < 20000; i++) {
      let now = new Date((base1 += oneDay1));
      let d = [+now, Math.abs(Math.round((Math.random() - 0.5) * 20 + data1[i - 1][1]))];
      data1.push(d);
    }
    var option1 = {
      grid: {
        top: "20%",
        
      },
      tooltip: {
        trigger: 'axis',
        position: function (pt) {
          return [pt[0], '10%'];
        }
      },
      title: {
        left: 'center',
        text: '灾情数量时间分布图',
        top: "5%"
      },
      toolbox: {
        feature: {
          restore: {},
          saveAsImage: {}
        }
      },
      xAxis: {
        type: 'time',
        boundaryGap: false
      },
      yAxis: {
        type: 'value',
        boundaryGap: [0, '100%']
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 20
        },
        {
          start: 0,
          end: 20
        }
      ],
      series: [
        {
          name: 'Fake Data',
          type: 'line',
          smooth: true,
          symbol: 'none',
          areaStyle: {},
          data: data1
        }
      ]
    };
    
    var option2 = {
      grid: {
        top: "25%"
      },
      title: {
        left: 'center',
        text: '灾情数量年份分布图',
        top: "5%"
      },
      xAxis: {
        type: 'category',
        data: ['2017', '2018', '2019', '2020', '2021', '2022', '2023']
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          data: [820, 932, 901, 934, 1290, 1330, 1320],
          type: 'line',
          smooth: true
        }
      ]
    };
    onMounted(() => {
      initChart();
    });

    onUnmounted(() => {
      echarts.dispose;
    });

    function initChart() {
      let chart1 = echarts.init(document.getElementById("chart1"), "purple-passion");
      let chart2 = echarts.init(document.getElementById("chart2"), "purple-passion");
      chart1.setOption(option1);
      chart2.setOption(option2);
      window.onresize = function () {
        chart1.resize();
      };
    }

    return {
      initChart,
      
    };
  }
};
</script>

