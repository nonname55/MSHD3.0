// 表单校验规则由 schema2code 生成，不建议直接修改校验规则，而建议通过 schema2code 生成, 详情: https://uniapp.dcloud.net.cn/uniCloud/schema


const validator = {
  "disasterCode": {
    "rules": [
      {
        "required": true
      },
      {
        "format": "string"
      }
    ],
    "title": "灾情编码",
    "label": "灾情编码"
  },
  "province": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "省份",
    "label": "省份"
  },
  "city": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "城市",
    "label": "城市"
  },
  "county": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "县/区",
    "label": "县/区"
  },
  "town": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "乡镇",
    "label": "乡镇"
  },
  "village": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "村/社区",
    "label": "村/社区"
  },
  "time": {
    "rules": [
      {
        "format": "timestamp"
      }
    ],
    "title": "时间",
    "label": "时间"
  },
  "source": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "来源",
    "label": "来源"
  },
  "subSource": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "来源子类",
    "label": "来源子类"
  },
  "carrier": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "载体",
    "label": "载体"
  },
  "category": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "灾情大类",
    "label": "灾情大类"
  },
  "subCategory": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "灾情子类",
    "label": "灾情子类"
  },
  "indicator": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "灾情指标",
    "label": "灾情指标"
  },
  "description": {
    "rules": [
      {
        "format": "string"
      }
    ],
    "title": "描述",
    "label": "描述"
  }
}

const enumConverter = {}

function filterToWhere(filter, command) {
  let where = {}
  for (let field in filter) {
    let { type, value } = filter[field]
    switch (type) {
      case "search":
        if (typeof value === 'string' && value.length) {
          where[field] = new RegExp(value)
        }
        break;
      case "select":
        if (value.length) {
          let selectValue = []
          for (let s of value) {
            selectValue.push(command.eq(s))
          }
          where[field] = command.or(selectValue)
        }
        break;
      case "range":
        if (value.length) {
          let gt = value[0]
          let lt = value[1]
          where[field] = command.and([command.gte(gt), command.lte(lt)])
        }
        break;
      case "date":
        if (value.length) {
          let [s, e] = value
          let startDate = new Date(s)
          let endDate = new Date(e)
          where[field] = command.and([command.gte(startDate), command.lte(endDate)])
        }
        break;
      case "timestamp":
        if (value.length) {
          let [startDate, endDate] = value
          where[field] = command.and([command.gte(startDate), command.lte(endDate)])
        }
        break;
    }
  }
  return where
}

export { validator, enumConverter, filterToWhere }
