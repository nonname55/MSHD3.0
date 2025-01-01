export default {
  path: "/ai",
  redirect: "/ai/classify",
  meta: {
    icon: "informationLine",
    title: "AI赋能",
    // showLink: false,
    rank: 9
  },
  children: [
    {
      path: "/ai/classify",
      name: "图片分类",
      component: () => import("@/views/ai/classify.vue"),
      meta: {
        title: "图片分类"
      }
    },
    {
      path: "/error/404",
      name: "404",
      component: () => import("@/views/error/404.vue"),
      meta: {
        title: "404"
      }
    },
    {
      path: "/error/500",
      name: "500",
      component: () => import("@/views/error/500.vue"),
      meta: {
        title: "500"
      }
    }
  ]
} as RouteConfigsTable;
