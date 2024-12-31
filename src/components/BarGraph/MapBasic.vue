<template>
  <button @click="changeVision" class="btn">切换视图</button>
  <div class="echarts-box">
    <div id="myEcharts" :style="{ width: width, height: height }"></div>
  </div>
</template>

<script>
import * as echarts from "echarts"; 
import { onMounted, onUnmounted, ref } from "vue";
import geoJson from "@/store/visual/China.json";
import { province_info } from "@/store/visual/provinceData.js";

export default {
  name: "App",
  props: ["width", "height"],
  setup() {
    let myEcharts = echarts;
    var provinces = province_info.slice();
    provinces.sort(function (a, b) {
      return a.value - b.value;
    });

    // 地图视图配置
    const mapOption = {
      layoutCenter: ['55%', '70%'], // 位置
      layoutSize: '100%', // 大小
      title: {
        text: '中国各省份灾害数量',
        left: "center",
        top: "5%"
      },
      tooltip: {
        trigger: 'item',
        formatter: '{b}\n{c}'
      },
      toolbox: {
        show: true,
        orient: 'vertical',
        left: 'right',
        top: 'center',
        feature: {
          dataView: { readOnly: false },
          restore: {},
          saveAsImage: {}
        }
      },
      visualMap: {
        min: 0,
        max: 1000,
        text: ['High', 'Low'],
        realtime: false,
        calculable: true,
        inRange: {
          color: ['lightskyblue', 'yellow', 'orangered']
        }
      },
      series: [
        {
          id: 'disaster',
          roam: true,
          animationDurationUpdate: 1000,
          universalTransition: true,
          name: '灾害数量',
          type: 'map',
          map: 'china',
          label: {
            show: true
          },
          data: provinces,
        }
      ]
    };

    // 条形图视图配置
    const barOption = {
      title: {
        text: '中国各省份灾害数量',
        left: "center",
        top: "5%"
      },
      xAxis: {
        type: 'value'
      },
      yAxis: {
        type: 'category',
        axisLabel: {
          rotate: 30
        },
        data: provinces.map(function (item) {
          return item.name;
        })
      },
      animationDurationUpdate: 1000,
      series: {
        type: 'bar',
        id: 'disaster',
        data: provinces.map(function (item) {
          return item.value;
        }),
        universalTransition: true
      }
    };

    // 当前选中的配置，默认是地图视图
    let currentOption = mapOption;

    onMounted(() => {
      initChart();
    });

    onUnmounted(() => {
      myEcharts.dispose();
    });

    function initChart() {
      let chart = myEcharts.init(document.getElementById("myEcharts"), "purple-passion");
      chart.hideLoading();
      echarts.registerMap('china', geoJson);
      chart.setOption(currentOption);
      chart.setOption(currentOption, true);
      window.onresize = function () {
        chart.resize();
      };
    }

    // 切换视图
    const changeVision = () => {
      currentOption = currentOption === mapOption ? barOption : mapOption;
      initChart();
    }

    return {
      initChart,
      province_info,
      geoJson,
      changeVision,
    };
  }
};
</script>

<style scoped>
/* 设定图表容器大小 */
.echarts-box {
  width: 100%;
  height: 400px;
}

/* 按钮样式 */
.btn {
  padding: 5px 15px;
  margin: 10px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.btn:hover {
  background-color: #0056b3;
}
</style>
