<template>
  <div>
    <view>
      <line-basic :width="'1200px'" :height="'600px'"></line-basic>
    </view>
  </div>
</template>

<script>
import LineBasic from "@/components/BarGraph/LineBasic.vue"
import PieBasic from "@/components/BarGraph/PieBasic.vue";

export default {
  name: "App",
  components: {
    LineBasic,
    PieBasic
  },
  data() {
    return {
      
    };
  },
};
</script>

<style>
</style>
