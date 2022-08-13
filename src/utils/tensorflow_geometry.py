import tensorflow as tf

def _build_quaternion_from_sines_and_cosines(
    sin_half_angles,
    cos_half_angles) -> tf.Tensor:
  """Builds a quaternion from sines and cosines of half Euler angles.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    sin_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the sine of half Euler angles.
    cos_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the cosine of half Euler angles.
  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a quaternion.
  """
  c1, c2, c3 = tf.unstack(cos_half_angles, axis=-1)
  s1, s2, s3 = tf.unstack(sin_half_angles, axis=-1)
  w = c1 * c2 * c3 + s1 * s2 * s3
  x = -c1 * s2 * s3 + s1 * c2 * c3
  y = c1 * s2 * c3 + s1 * c2 * s3
  z = -s1 * s2 * c3 + c1 * c2 * s3
  return tf.stack((x, y, z, w), axis=-1)
  
def quat_from_euler(angles,
               name: str = "quaternion_from_euler"
               ) -> tf.Tensor:
  """Converts an Euler angle representation to a quaternion.
  Note:
    Uses the z-y-x rotation convention (Tait-Bryan angles).
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[..., 0]` is the angle about `x` in
      radians, `[..., 1]` is the angle about `y` in radians and `[..., 2]` is
      the angle about `z` in radians.
    name: A name for this op that defaults to "quaternion_from_euler".
  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.
  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)

    half_angles = angles / 2.0
    cos_half_angles = tf.cos(half_angles)
    sin_half_angles = tf.sin(half_angles)
    return _build_quaternion_from_sines_and_cosines(sin_half_angles,
                                                    cos_half_angles)
def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles) -> tf.Tensor:
  """Builds a rotation matrix from sines and cosines of Euler angles.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the sine of the Euler angles.
    cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the cosine of the Euler angles.
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  """
  sx, sy, sz = tf.unstack(sin_angles, axis=-1)
  cx, cy, cz = tf.unstack(cos_angles, axis=-1)
  m00 = cy * cz
  m01 = (sx * sy * cz) - (cx * sz)
  m02 = (cx * sy * cz) + (sx * sz)
  m10 = cy * sz
  m11 = (sx * sy * sz) + (cx * cz)
  m12 = (cx * sy * sz) - (sx * cz)
  m20 = -sy
  m21 = sx * cy
  m22 = cx * cy
  matrix = tf.stack((m00, m01, m02,
                     m10, m11, m12,
                     m20, m21, m22),
                    axis=-1)  # pyformat: disable
  output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
  return tf.reshape(matrix, shape=output_shape)
def mat_from_euler(angles,
               name: str = "rotation_matrix_3d_from_euler") -> tf.Tensor:
  r"""Convert an Euler angle representation to a rotation matrix.
  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)
    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def _build_rot_matrix_from_sines_and_cosines(sin_angles, cos_angles) -> tf.Tensor:
  """Builds a rotation matrix from sines and cosines of Euler angles.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the sine of the Euler angles.
    cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the cosine of the Euler angles.
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  """
  sx, sy, sz = tf.unstack(sin_angles, axis=-1)
  cx, cy, cz = tf.unstack(cos_angles, axis=-1)
  id1 = tf.ones_like(sx)
  id0 = tf.zeros_like(sx)
  matrix = tf.stack((id1, id0, -sy,
                     id0,  cx, sx*cy,
                     id0, -sx, cx*cy),
                    axis=-1)  # pyformat: disable
#   matrix = tf.stack((id1, id0,  id0,
#                      id0, cx, -sx,
#                      m20, m21, m22),
#                     axis=-1)  # pyformat: disable
  output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
  return tf.reshape(matrix, shape=output_shape)
def rot_mat_from_euler(angles,
               name: str = "rotation_matrix_3d_from_euler") -> tf.Tensor:
  r"""Convert an Euler angle representation to a rotation matrix.
  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)
    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    return _build_rot_matrix_from_sines_and_cosines(sin_angles, cos_angles)

