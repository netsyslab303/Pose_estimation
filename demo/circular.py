class CircularQueue:

    # 큐 초기화
    def __init__(self, n):
        self.maxCount = n
        self.data = [None] * n
        self.count = 0
        self.front = -1
        self.rear = -1

    # 현재 큐 길이를 반환
    def size(self):
        return self.count

    # 큐가 비어있는지
    def isEmpty(self):
        return self.count == 0

    # 큐가 꽉 차있는지
    def isFull(self):
        return self.count == self.maxCount

    # 데이터 원소 추가
    def enqueue(self, x):
        if self.isFull():
            raise IndexError('Queue full')

        self.rear = (self.rear + 1) % self.maxCount
        self.data[self.rear] = x
        self.count += 1

    # 데이터 원소 제거
    def dequeue(self):
        if self.isEmpty():
            raise IndexError('Queue empty')

        self.front = (self.front + 1) % self.maxCount
        self.data[self.front] = None
        self.count -= 1

    # 큐의 맨 앞 원소 반환
    def peek(self):
        if self.isEmpty():
            raise IndexError('Queue empty')
        return self.data[(self.front + 1) % self.maxCount]

    #해당 원소가 있는지  확인
    def search_que(self, x):
        for i in range(self.size()):
            tmp2 = abs(((self.front + i + 1) % self.maxCount) - ((self.front+ 1) % self.maxCount))

            if self.data[(self.front + i + 1) % self.maxCount][1] == x:
                return i % self.maxCount

        return None



    #x번째 데이터 반환
    def value(self, x):

        # if self.data[(self.front + x + 1) % self.maxCount] is None:
        #     raise IndexError('Queue empty')
        return self.data[(self.front + x + 1) % self.maxCount]

    # 현재 가장 앞 데이터의 idx 값 수정
    def value_modify(self,loc, index, data):
        if self.isEmpty():
            raise IndexError('Queue empty')
        self.data[(self.front + loc + 1) % self.maxCount][index] = data

    # 원하는 위치의 데이터 idx 값에 data 더한다.
    def value_add(self, loc ,index, data):
        if self.isEmpty():
            raise IndexError('Queue empty')
        self.data[(self.front + loc + 1) % self.maxCount][index] += data

    def print_queue(self):
        if self.isEmpty():
            print('Queue empty')
        x = []
        for i in range(0, self.size()):
            x.append(self.data[(self.front + i +1) % self.maxCount])
        print(x)

    def max_len(self):
        max_len = 0
        if self.isEmpty():
            print('Queue empty')
        for i in range(0, self.size()):
            length = len(self.data[(self.front + i +1) % self.maxCount])
            if  length > max_len:
                max_len = length
        return max_len

def test(input):
    input.dequeue()


# uav_task = [CircularQueue(10) for j in range(3)]
# for i in range (0,4):
#     uav_task[0].enqueue(i*10)
#     uav_task[0].enqueue(i*10+1)
#     uav_task[1].enqueue("20")
#
# uav_task[0].dequeue()
# test(uav_task[0])
# uav_task[0].enqueue(60)
# uav_task[0].enqueue(70)
# uav_task[0].enqueue(80)
# uav_task[0].enqueue(90)
#
# #test(uav_task[0])
#
# uav_task[0].print_queue()
