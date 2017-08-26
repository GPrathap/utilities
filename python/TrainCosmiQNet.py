with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(ginit)
    global_step = 0
    for i in range(layers):
        for step in range(FLAGS.max_layer_steps):
            global_step += 1
            random_input8, random_input3, random_label = GetRandomInput( labels, im_raw8, im_raw3, FLAGS.batch_size, FLAGS.ws, scale)
            sess.run(label_optimizer[i],feed_dict={x8: random_input8, x3: random_input3, label_distance: random_label})
        for step in range(5*FLAGS.max_layer_steps):
            global_step += 1
            random_input8, random_input3, random_label = GetRandomInput( labels, im_raw8, im_raw3, FLAGS.batch_size, FLAGS.ws, scale)
            sess.run(full_label_optimizer[i],feed_dict={x8: random_input8, x3: random_input3, label_distance: random_label})
        gsaver.save(sess, './checkpoint/boundary_'+str(i))
    gsaver.save(sess, './checkpoint/boundary')