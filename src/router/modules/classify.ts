export default {
  path: "/classify",
  redirect: "/classify/image",
  meta: {
    icon: "informationLine",
    title: "AI赋能",
    // showLink: false,
    rank: 9
  },
  children: [
    {
      path: "/classify/image",
      name: "图片分类",
      component: () => import("@/views/classify/image.vue"),
      meta: {
        title: "图片分类"
      }
    },
    {
      path: "/classify/tweet",
      name: "推文分类",
      component: () => import("@/views/classify/tweet.vue"),
      meta: {
        title: "推文分类"
      }
    },
  ]
} as RouteConfigsTable;
