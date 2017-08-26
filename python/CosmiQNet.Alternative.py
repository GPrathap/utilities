with tf.device(gpu):
# Generator
    x8 = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 8])
    x3 = tf.placeholder(tf.float32, shape=[None, scale * FLAGS.ws, scale * FLAGS.ws, 3])
    label_distance = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 1])
    for i in range(layers):
        alpha[i] = tf.Variable(0.9, name='alpha_' + str(i))
        beta[i] = tf.maximum( 0.0 , tf.minimum ( 1.0 , alpha[i] ), name='beta_'+str(i))
        bi[i] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='bi_'+str(i))
        bo[i] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='bo_'+str(i))
        Wo[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,1,FLAGS.filters], stddev=0.1), name='Wo_'+str(i))  #
        Wi3[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,3,FLAGS.filters], stddev=0.1), name='Wi_'+str(i)+'l3')
        Wi8[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,8,FLAGS.filters], stddev=0.1), name='Wi_'+str(i)+'l8')
        z3[i] = tf.nn.conv2d( x3, Wi3[i], strides=[1,scale,scale,1], padding='SAME')
        z8[i] = tf.nn.conv2d( x8, Wi8[i], strides=[1,1,1,1], padding='SAME')
        if 0 == i:
            z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(tf.add(z3[i], z8[i]), bi[i], name='conv_'+str(i))), bo[i])
        else:
            inlayer[i] = outlayer[i-1]
            Wi[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,1,FLAGS.filters], stddev=0.1), name='Wi_'+str(i))
            z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(
                       tf.add(tf.add(z3[i],z8[i]),tf.nn.conv2d( inlayer[i], Wi[i], strides=[1,1,1,1], padding='SAME')),
                       bi[i], name='conv_'+str(i))), bo[i])            
        Wii[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,FLAGS.filters,FLAGS.filters], stddev=0.1), name='Wii_'+str(i))
        bii[i] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='bii_'+str(i))
        zz[i] = tf.nn.relu( tf.nn.bias_add( tf.nn.conv2d( z[i], Wii[i], strides=[1,1,1,1], padding='SAME'), bii[i]))
        labelout[i] = tf.nn.conv2d_transpose( zz[i], Wo[i], [FLAGS.batch_size,FLAGS.ws,FLAGS.ws,1] ,strides=[1,1,1,1], padding='SAME')
        if 0 == i:
            outlayer[i] = labelout[i]
        else :
            outlayer[i] = tf.nn.relu( tf.add(  tf.scalar_mul( beta[i] , labelout[i]), tf.scalar_mul(1.0-beta[i], inlayer[i])))
        label_cost[i] = tf.reduce_sum ( tf.pow( tf.sub(outlayer[i],label_distance),2))