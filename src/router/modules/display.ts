// 最简代码，也就是这些字段必须有
export default {
  path: "/display",
  meta: {
    title: "展示"
  },
  children: [
    {
      path: "/display/index",
      name: "Display",
      component: () => import("@/views/display/index.vue"),
      meta: {
        title: "详细信息"
      }
    },
    {
      path: "/display/pie",
      name: "DisplayPie",
      component: () => import("@/views/display/pie.vue"),
      meta: {
        title: "饼图"
      }
    },
    {
      path: "/display/map",
      name: "DisplayMap",
      component: () => import("@/views/display/map.vue"),
      meta: {
        title: "地图视图"
      }
    },
    {
      path: "/display/barrace",
      name: "DisplayBarRace",
      component: () => import("@/views/display/bar_race.vue"),
      meta: {
        title: "动态柱状图"
      }
    },
    {
      path: "/display/heat",
      name: "DisplayHeat",
      component: () => import("@/views/display/heat.vue"),
      meta: {
        title: "日历热力图"
      }
    },
    {
      path: "/display/line",
      name: "DisplayLine",
      component: () => import("@/views/display/line.vue"),
      meta: {
        title: "折线图"
      }
    }
  ]
};
