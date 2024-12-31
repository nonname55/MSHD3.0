import Mock from "mockjs";

// 定义生成数据的模板
const dataTemplate = {
  // 生成10条数据
  id: /[0-9]{26}/, // 使用正则表达式生成26位数字字符串
  location: "@county(true)", // 使用 Mock.js 的随机地址
  time: "@datetime", // 随机日期时间
  source: "@word(3, 5)", // 随机字符串作为数据源
  file: "@url", // 随机URL
  category: "@word(3, 5)", // 随机分类
  tags: "@word(3, 5)", // 随机标签
  description: "@sentence(5,10)", // 随机描述
  sourceUrl:
    "https://img.zcool.cn/community/01471c5791780e0000012e7ebf7ec0.png@1280w_1l_2o_100sh.png"
};

// 模拟地震数据
Mock.mock("/api/earthquakes", "get", {
  code: 200,
  "data|50": [dataTemplate],
  message: "请求成功"
});
Mock.mock(RegExp("/api/earthquakes/search" + ".*"), "get", function (options) {
  // 生成测试数据
  const generatedData = Mock.mock({
    "array|50": [dataTemplate]
  }).array;

  // 解析 URL 中的查询字符串
  const queryParams = new URLSearchParams(options.url.split("?")[1]);
  const searchKey = queryParams.get("searchKey");

  // 如果没有提供 searchKey 或 searchKey 为空，则返回所有数据
  if (!searchKey) {
    return {
      code: 200,
      data: generatedData,
      message: "请求成功"
    };
  }

  // 根据 searchKey 进行过滤
  const filteredData = generatedData.filter(
    item =>
      item.id.includes(searchKey) ||
      item.location.includes(searchKey) ||
      item.time.includes(searchKey) ||
      item.source.includes(searchKey) ||
      item.file.includes(searchKey) ||
      item.category.includes(searchKey) ||
      item.tags.includes(searchKey) ||
      item.description.includes(searchKey)
  );

  return {
    code: 200,
    data: filteredData,
    message: "请求成功"
  };
});
