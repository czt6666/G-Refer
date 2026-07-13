# GraphLoRA

TALLRec把用户和物品的交互历史转为文字

CoLLM把用户和物品嵌入成软token

但这只能让LLM读结构信息，不能让llm把结构信息训练进去

而且token/线性化的关系不能表示高阶信息

GraphLoRA是在lora中加入了GNN，让图结构和lora一起训练











