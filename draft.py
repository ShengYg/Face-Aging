from tensorflow.contrib.training.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.ops import init_ops

def gan_train_ops(
    self,
    model,
    loss,
    generator_optimizer,
    discriminator_optimizer,
    check_for_unused_update_ops=True,
):


    # Create global step increment op.
    global_step = training_util.get_or_create_global_step()
    global_step_inc = global_step.assign_add(1)

    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    all_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, model.EG_scope.name))
    update_ops = list(all_ops & update_ops)
    with ops.name_scope('EG_train'):
        gen_train_op = training.create_train_op(
            total_loss=self.loss_EG,
            optimizer=self.EG_optimizer,
            variables_to_train=self.E_variables + self.G_variables,
            global_step=self.EG_global_step,
            update_ops=update_ops)

    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    all_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, model.Dz_scope.name))
    update_ops = list(all_ops & update_ops)
    with ops.name_scope('Dz_train'):
        gen_train_op = training.create_train_op(
            total_loss=self.loss_Dz,
            optimizer=self.D_z_optimizer,
            variables_to_train=self.E_variables + self.G_variables,
            global_step=self.EG_global_step,
            update_ops=update_ops)

    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    all_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, model.Di_scope.name))
    update_ops = list(all_ops & update_ops)
    with ops.name_scope('Di_train'):
        gen_train_op = training.create_train_op(
            total_loss=self.loss_Di,
            optimizer=self.D_img_optimizer,
            variables_to_train=self.E_variables + self.G_variables,
            global_step=self.EG_global_step,
            update_ops=update_ops)

    return namedtuples.GANTrainOps(gen_train_op, disc_train_op, global_step_inc)




def gan_train(
    self,
    train_ops,
    logdir,
    get_hooks_fn=get_joint_train_hooks(),
    master='',
    is_chief=True,
    scaffold=None,
    hooks=None,
    chief_only_hooks=None,
    save_checkpoint_secs=600,
    save_summaries_steps=100,
    config=None
):
    new_hooks = get_hooks_fn(train_ops)
    if hooks is not None:
        hooks = list(hooks) + list(new_hooks)
    else:
        hooks = new_hooks
    return training.train(
            train_ops.global_step_inc_op,
            logdir,
            master=master,
            is_chief=is_chief,
            scaffold=scaffold,
            hooks=hooks,
            chief_only_hooks=chief_only_hooks,
            save_checkpoint_secs=save_checkpoint_secs,
            save_summaries_steps=save_summaries_steps,
            config=config)


def get_joint_train_hooks(self, train_steps=(1, 1)):
    g_steps = train_steps[0]
    d_steps = train_steps[1]
    # Get the number of each type of step that should be run.
    num_d_and_g_steps = min(g_steps, d_steps)
    num_g_steps = g_steps - num_d_and_g_steps
    num_d_steps = d_steps - num_d_and_g_steps

    def get_hooks(train_ops):
        g_op = train_ops.generator_train_op
        d_op = train_ops.discriminator_train_op

        joint_hook = RunTrainOpsHook([g_op, d_op], num_d_and_g_steps)
        g_hook = RunTrainOpsHook(g_op, num_g_steps)
        d_hook = RunTrainOpsHook(d_op, num_d_steps)

        return [joint_hook, g_hook, d_hook]
    return get_hooks






status_message = tf.string_join(
        ['Starting train step: ', 
        tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')

gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
         tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
    logdir=FLAGS.train_log_dir,
    get_hooks_fn=get_joint_train_hooks())