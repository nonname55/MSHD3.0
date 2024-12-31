<template>
  <div class="echarts-box">
    <div id="myEcharts" :style="{ width: this.width, height: this.height }"></div>
  </div>
</template>

<script>
import * as echarts from "echarts";
import {onMounted, onUnmounted} from "vue";

export default {
  name: "App",
  props: ["width", "height"],
  setup() {
    let myEcharts = echarts;
    
    onMounted(() => {
      var chartDom = document.getElementById('myEcharts');
      var myChart = echarts.init(chartDom);
      var option;
      
      const data = [];
      for (let i = 0; i < 5; ++i) {
        data.push(Math.round(Math.random() * 200));
      }
      option = {
        title: {
          text: "2010~2023年中国主要灾害动态柱状图",
          left: "center",
          top: "10%"
        },
        grid: {
          top: "20%",
          
        },
        xAxis: {
          max: 'dataMax',
          min: 'dataMin',
        },
        yAxis: {
          type: 'category',
          data: ['洪涝', '地震', '台风', '火灾', '干旱'],
          inverse: true,
          animationDuration: 300,
          animationDurationUpdate: 300,
          max: 4
        },
        series: [
          {
            realtimeSort: true,
            name: 'X',
            type: 'bar',
            data: data,
            label: {
              show: true,
              position: 'right',
              valueAnimation: true
            }
          }
        ],
        animationDuration: 0,
        animationDurationUpdate: 3000,
        animationEasing: 'linear',
        animationEasingUpdate: 'linear'
      };
      function run() {
        for (var i = 0; i < data.length; ++i) {
          if (Math.random() > 0.9) {
            data[i] += Math.round(Math.random() * 2000);
          } else {
            data[i] += Math.round(Math.random() * 200);
          }
        }
        myChart.setOption({
          series: [
            {
              type: 'bar',
              data
            }
          ]
        });
      }
      setTimeout(function () {
        run();
      }, 0);
      setInterval(function () {
        run();
      }, 3000);
      
      option && myChart.setOption(option);
    });

    onUnmounted(() => {
      echarts.dispose;
    });

    return {
    
    };
  }
};
</script>

