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
    
    function getVirtualData(year) {
      const date = +echarts.time.parse(year + '-01-01');
      const end = +echarts.time.parse(+year + 1 + '-01-01');
      const dayTime = 3600 * 24 * 1000;
      const data = [];
      for (let time = date; time < end; time += dayTime) {
        data.push([
          echarts.time.format(time, '{yyyy}-{MM}-{dd}', false),
          Math.floor(Math.random() * 1000)
        ]);
      }
      return data;
    }
    
    let option = {
      title: {
        text: "最近三年中国灾害数量日历热力图",
        left: "center",
        top: "5%"
      },
      tooltip: {
        position: 'top',
      },
      visualMap: {
        min: 0,
        max: 1000,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        top: 100
      },
      calendar: [
        {
          top: 190,
          range: '2017',
          cellSize: ['auto', 20]
        },
        {
          top: 380,
          range: '2016',
          cellSize: ['auto', 20]
        },
        {
          top: 570,
          range: '2015',
          cellSize: ['auto', 20],
          right: 5
        }
      ],
      series: [
        {
          type: 'heatmap',
          coordinateSystem: 'calendar',
          calendarIndex: 0,
          data: getVirtualData('2017')
        },
        {
          type: 'heatmap',
          coordinateSystem: 'calendar',
          calendarIndex: 1,
          data: getVirtualData('2016')
        },
        {
          type: 'heatmap',
          coordinateSystem: 'calendar',
          calendarIndex: 2,
          data: getVirtualData('2015')
        }
      ]
    };

    onMounted(() => {
      initChart();
    });

    onUnmounted(() => {
      myEcharts.dispose;
    });

    function initChart() {
      let chart = myEcharts.init(document.getElementById("myEcharts"), "purple-passion");
      chart.setOption(option)
      window.onresize = function () {
        chart.resize();
      };
    }

    return {
      initChart
    };
  }
};
</script>

