import tensorflow as tf
X = [1,2,3] #입력값 matrix
Y = [5,8,11] #결괏값 matrix
W = tf.Variable(5.0) # W의 값이 5로 시작하는 변수. optimize 된 값은 1라서 빠르게 1로 수렴함
b = tf.Variable(3.0)
hypothesis = X*W+b
cost = tf.reduce_mean(tf.square(hypothesis-Y)) #cost의 식을 그대로 사용한거다! (예상값-y)^2의 평균값!
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1) #경사 하강법
train = optimizer.minimize(cost) 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(250):
    print(cost, step,sess.run(W), sess.run(b)) # W의 값이 3, b의 값이 2로 수렴하는 것을 볼 수 있다!
    sess.run(train)
