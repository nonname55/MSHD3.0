<template>
  <div>
    <header class="header">
      <nav class="breadcrumb">
        <span>省份数据热力图</span>
      </nav>
    </header>
    <div id="mapContainer"></div>
    <!-- 添加调试信息显示 -->
    <div class="debug-info" v-if="debug">
      <p>最大值: {{ maxValue }}</p>
      <div v-for="(value, code) in provinceValues" :key="code">
        省份代码: {{ code }}, 值: {{ value }}, 颜色强度: {{ value / maxValue }}
      </div>
    </div>
  </div>
</template>

<script>
import AMapLoader from '@amap/amap-jsapi-loader'

// 省份数据
const province_info = [
  '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', '上海市', '江苏省', 
  '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省', '广东省', '广西壮族自治区', 
  '海南省', '重庆市', '四川省', '贵州省', '云南省', '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', 
  '新疆维吾尔自治区', '台湾省', '香港特别行政区', '澳门特别行政区'
].map(province => ({
  value: Math.floor(Math.random() * 1000),
  name: province
}));

// 省份名称到行政区划代码的映射
const provinceToAdcode = {
  '北京市': 110000, '天津市': 120000, '河北省': 130000, '山西省': 140000,
  '内蒙古自治区': 150000, '辽宁省': 210000, '吉林省': 220000, '黑龙江省': 230000,
  '上海市': 310000, '江苏省': 320000, '浙江省': 330000, '安徽省': 340000,
  '福建省': 350000, '江西省': 360000, '山东省': 370000, '河南省': 410000,
  '湖北省': 420000, '湖南省': 430000, '广东省': 440000, '广西壮族自治区': 450000,
  '海南省': 460000, '重庆市': 500000, '四川省': 510000, '贵州省': 520000,
  '云南省': 530000, '西藏自治区': 540000, '陕西省': 610000, '甘肃省': 620000,
  '青海省': 630000, '宁夏回族自治区': 640000, '新疆维吾尔自治区': 650000, '台湾省': 710000,
  '香港特别行政区': 810000, '澳门特别行政区': 820000
};

window._AMapSecurityConfig = {
  securityJsCode: 'b64ab7ba784b4ab4852e272f08fc19df'
}

export default {
  name: 'ProvinceMap',
  data() {
    return {
      map: null,
      disProvince: null,
      adcodeList: [],
      provinceValues: {},
      maxValue: 0,
      debug: true // 调试模式开关
    }
  },
  created() {
    this.initData()
  },
  methods: {
    initAMap() {
      AMapLoader.load({
        key: "56ea5c53fce4ac4c13c9efbe91a23884",
        version: "2.0",
        plugins: ['AMap.DistrictLayer']
      })
      .then((AMap) => {
        this.map = new AMap.Map("mapContainer", {
          viewMode: "3D",
          resizeEnable: true,
          zoom: 4.2,
          center: [108.95351, 38.26562]
        });
        
        this.disProvince = new AMap.DistrictLayer.Province({
          zIndex: 12,
          adcode: this.adcodeList,
          depth: 2,
          styles: {
            'fill': (props) => {
              const adcode = props.adcode;
              // 使用字符串截取获取省份代码
              const provinceCode = parseInt(String(adcode).substring(0, 2));
              const fullCode = provinceCode * 10000;
              const value = this.provinceValues[fullCode] || 0;
              
              // 计算颜色强度，确保值在0.1到0.9之间
              const intensity = 0.1 + (value / this.maxValue) * 0.8;
              
              // 使用console.log记录每个省份的颜色计算
              //console.log(`Province ${adcode}: value=${value}, intensity=${intensity}`);
              
              return `rgba(255, 0, 0, ${intensity})`;
            },
            'province-stroke': 'cornflowerblue',
            'city-stroke': 'white',
            'county-stroke': 'rgba(255,255,255,0.5)'
          }
        });
        
        this.disProvince.setMap(this.map);
      })
      .catch((e) => {
        console.error('地图加载错误:', e);
      });
    },

    initData() {
      // 初始化省份数据
      province_info.forEach(province => {
        const adcode = provinceToAdcode[province.name];
        if (adcode) {
          this.adcodeList.push(adcode);
          this.provinceValues[adcode] = province.value;
        }
      });

      // 计算最大值
      this.maxValue = Math.max(...Object.values(this.provinceValues));

      // console.log('初始化数据:', {
      //   provinceValues: this.provinceValues,
      //   maxValue: this.maxValue,
      //   adcodeList: this.adcodeList
      // });

      this.initAMap();
    }
  },
  beforeDestroy() {
    if (this.map) {
      this.map.destroy();
    }
  }
}
</script>

<style scoped>
#mapContainer {
  padding: 0;
  margin: 0;
  width: 100%;
  height: 600px;
}

.header {
  padding: 16px;
  background-color: #f5f5f5;
}

.breadcrumb {
  font-size: 16px;
  color: #333;
  margin-bottom: 10px;
}

.debug-info {
  padding: 16px;
  background-color: #f5f5f5;
  margin-top: 16px;
  font-size: 12px;
}
</style>