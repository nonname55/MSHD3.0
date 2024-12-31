<template>
  <view>
    <calender-heat :width="'1200px'" :height="'800px'"></calender-heat>
  </view>
</template>

<script>
import CalenderHeat from "@/components/BarGraph/CalenderHeat.vue"

export default {
  name: "App",
  components: {
    CalenderHeat
  },
  data() {
    return {
      
    };
  },
};
</script>

<style>
</style>
