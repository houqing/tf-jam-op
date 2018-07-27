#!/usr/bin/env python3

import os

import tensorflow as tf

jam_me_module = tf.load_op_library("jamme-op.so")

with tf.device('/device:GPU:0'):
    inp = tf.constant([10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0., -1., -2., -3., -4., -5., -6., -7., -8., -9., -10.], dtype=tf.float16)
    jam, inp_raw, sim = jam_me_module.jam_me(inp)



sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
if False:
    result = tf.Print(jam, [jam, inp_raw, inp, sim], message="result (jam, inp_raw, inp, sim) : ", summarize=9999)
    sess.run(result)
else:
    print("in ", inp.eval(session=sess))
    print("jam", jam.eval(session=sess))
    print("ran", inp_raw.eval(session=sess))
    print("sim", sim.eval(session=sess))

