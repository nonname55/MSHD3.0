<template>
  <div class="container">
    <div class="chart-container">
      <div id="myEcharts" :style="{ width: width, height: height }"></div>
    </div>
    <div class="table-container">
      <table class="custom-table" border="1">
        <!-- 表头行 -->
        <thead>
          <tr>
            <th align="center">省份</th>
            <th align="center">灾害数量</th>
          </tr>
        </thead>
        <!-- 表格数据行 -->
        <tbody>
          <tr v-for="item in lastData" :key="item.name">
            <td>{{ item.name }}</td>
            <td>{{ item.value }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  <div class="button-container">
    <button @click="selectAll" class="btn">全选</button>
    <button @click="unselectAll" class="btn">全不选</button>
  </div>
</template>

<script>
import { onMounted, onUnmounted, ref } from "vue";
import * as echarts from "echarts";
import { province_info } from "@/store/visual/provinceData.js"

export default {
  name: "App",
  props: ["width", "height"],
  setup() {
    const myEcharts = ref(null);
    const lastData = ref(province_info.map(item => ({ ...item })));

    onMounted(() => {
      initChart();
      myEcharts.value.on('legendselectchanged', function (params) {
        var selected = params.selected;
        updateLastData(selected);
      });
    });
    
    const updateLastData = (newSelected) => {
      var newSelectedData = [];
      for (var key in newSelected) {
        if (newSelected[key]) {
          newSelectedData.push(province_info.find(item => item.name === key));
        } else {
          var index = lastData.value.findIndex(item => item.name === key);
          if (index !== -1) {
            lastData.value.splice(index, 1);
          }
        }
      }
      lastData.value = newSelectedData;
    };

    onUnmounted(() => {
      myEcharts.value.dispose();
    });

    function initChart() {
      myEcharts.value = echarts.init(document.getElementById("myEcharts"), "purple-passion");
      const legendData = province_info.map(info => info.name);
      myEcharts.value.setOption({
        tooltip: {
          trigger: 'item', 
          formatter: '{a} \n{b} : {c} ({d}%)'
        },
        title: {
          text: "各省份灾情数量统计图",
          left: "center",
          top: "10%"
        },
        legend: {
          orient: 'vertical',
          right: 0,
          top: '20%',
          left: '70%',
          data: legendData,
        },
        series: [
          {
            name: '灾情数量（百分比）',
            type: 'pie',
            data: province_info,
            radius: '80%',
            top: '20%',
            center: ['35%', '50%']
          },
        ],
      });
      window.onresize = function () {
        myEcharts.value.resize();
      };
    }
    
    const selectAll = () => {
      const selectedData = province_info.map(info => {
        return {
          name: info.name,
          selected: true
        };
      });
    
      myEcharts.value.setOption({
        legend: {
          selected: selectedData
        }
      });
      lastData.value = province_info.map(item => ({ ...item }));
    };
    const unselectAll = () => {
      const selectedData = {};
      province_info.forEach(info => {
        selectedData[info.name] = false;
      });
      myEcharts.value.setOption({
        legend: {
          selected: selectedData
        }
      });
      lastData.value = [];
    };

    return {
      province_info,
      selectAll,
      unselectAll,
      lastData,
      updateLastData,
    };
  }
};
</script>

<style scoped>
.container {
  display: flex;
}

.chart-container {
  margin-right: 10px;
}

.table-container {
  overflow: auto;
  max-height: 500px;
  margin-top: 100px;
}

.custom-table {
  width: 100%;
  border-collapse: collapse;
}

.custom-table th,
.custom-table td {
  padding: 10px;
  text-align: center;
  border: 1px solid #ddd;
}

.button-container {
  display: flex;
  flex-direction: row;
  margin-top: 10px;
}

.btn {
  padding: 5px 15px;
  margin: 5px;
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
