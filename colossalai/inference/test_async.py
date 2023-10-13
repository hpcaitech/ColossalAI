import asyncio

shared_list = []

async def producer():
    for i in range(5):
        await asyncio.sleep(1)  # 模拟异步获取数据的操作
        shared_list.append(i)
        print(f"Produced {i}")

async def consumer():
    last_index = 0
    while True:
        await asyncio.sleep(0.5)  # 为了不使循环过于紧凑，增加了小的延迟
        if last_index < len(shared_list):
            item = shared_list[last_index]
            print(f"Consumed {item}")
            yield item
            last_index += 1

async def main():
    # 创建生产者和消费者任务
    prod_task = asyncio.create_task(producer())
  
    # 等待生产者任务完成
    await prod_task

    async for  data in consumer():
        print(data)
    # 为了示例的目的，我们只等待一段时间，然后停止消费者
    await asyncio.sleep(5)

asyncio.run(main())
