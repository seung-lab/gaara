from typing import Union

import numpy as np
import numpy.typing as npt

from crackle import CrackleArray
from osteoid import Skeleton, Bbox
import xs3d

try:
  import fastremap
  unique = fastremap.unique
except ImportError:
  unique = np.unique

__all__ = [
    "cross_sectional_area",
]

def cross_sectional_area(
  all_labels:np.ndarray, 
  skeletons:Union[dict[int,Skeleton],list[Skeleton],Skeleton],
  anisotropy:np.ndarray = np.array([1,1,1], dtype=np.float32),
  smoothing_window:int = 5,
  progress:bool = False,
  multipass:bool = False,
  repair_contacts:bool = False,
  visualize_section_planes:bool = False,
  step:int = 1,
) -> Union[dict[int,Skeleton],list[Skeleton],Skeleton]:
  """
  Given a set of skeletons, find the cross sectional area
  for each vertex indicated by the sectioning plane
  defined by the vector pointing to the next vertex.

  When the smoothing_window is >1, these plane normal 
  vectors will be smoothed with a rolling average. This
  is useful since there can be high frequency
  oscillations in the skeleton.

  This function will add the following attributes to
  each skeleton provided.

  skel.cross_sectional_area: float32 array of cross 
    sectional area per a vertex.

  skel.cross_sectional_area_contacts: uint8 array
    where non-zero entries indicate that the image
    border was contacted during the cross section
    computation, indicating a possible underestimate.

    The first six bits are a bitfield xxyyzz that
    tell you which image faces were touched and
    alternate from low (0) to high (size-1).

  multipass: When True, preserve existing cross_sectional_area
    and contact values and allow values with zero to be recalculated.
    This is useful for example, when using a large skeleton with
    different sections of an image. Very similar to repair_contacts,
    except that any vertex can be considered, not just contacts.

  repair_contacts: When True, only examine vertices
    that have a nonzero value for 
    skel.cross_sectional_area_contacts. This is intended
    to be used as a second pass after widening the image.

  visualize_section_planes: For debugging, paint section planes
    and display them using microviewer.

  step: when > 1, skip (step-1) vertices. This can be used to
    go faster. These days, evaluating a single vertex takes 
    between a few hundred microseconds to a few thousand microseconds.
      example calculation: 
      1 msec x 100,000 vertices = 100 sec
      A neuron I recently examined had over 300,000 vertices across 
      the entire dataset.
      Kimimaro's benchmark task produced 622,293 vertices over 1667 objects 
      using reasonable parameters and took a little over 4 minutes on an M3 
      processor (or about 2.5 msec/vertex). The most expensive shape was the soma.
  """
  assert step > 0
  assert smoothing_window > 0

  def cross_sectional_area_helper(skel, binimg, roi, segid=None):
    cross_sections = None
    if visualize_section_planes:
      cross_sections = np.zeros(binimg.shape, dtype=np.uint32, order="F")

    if skel.space == "physical":
      all_verts = (skel.vertices / anisotropy).round().astype(int)
    else:
      all_verts = np.copy(skel.vertices)

    all_verts -= roi.minpt

    mapping = { tuple(v): i for i, v in enumerate(all_verts) }

    visited = np.zeros([ all_verts.shape[0] ], dtype=bool)

    if repair_contacts or (multipass and hasattr(skel, "cross_sectional_area")):
      areas = skel.cross_sectional_area
      contacts = skel.cross_sectional_area_contacts
    else:
      areas = np.zeros([all_verts.shape[0]], dtype=np.float32)
      contacts = np.zeros([all_verts.shape[0]], dtype=np.uint8)

    branch_pts = set(skel.branches())
    branch_pt_vals = defaultdict(list)

    paths = skel.paths()

    normal = np.array([1,0,0], dtype=np.float32)

    shape = np.array(binimg.shape)

    for path in paths:
      if skel.space == "physical":
        path = (path / anisotropy).round().astype(int)
      path -= roi.minpt

      normals = (path[1:] - path[:-1]).astype(np.float32)
      normals = np.concatenate([ normals, [normals[-1]] ])

      # Running the filter in the forward and then backwards
      # direction eliminates phase shift.
      normals = moving_average(normals, smoothing_window)
      normals = moving_average(normals[::-1], smoothing_window)[::-1]

      normals /= np.linalg.norm(normals, axis=1, keepdims=True)   

      end_i = len(path) - 1
      ct = 0

      for i, vert in enumerate(path):
        ct += 1

        if ct < step and not (i == 0 or i == end_i):
          continue
        elif ct == step:
          ct = 0

        if ( 
             (vert[0] < 0) 
          or (vert[0] >= shape[0])
          or (vert[1] < 0) 
          or (vert[1] >= shape[1])
          or (vert[2] < 0) 
          or (vert[2] >= shape[2])
        ):
          continue

        idx = mapping[tuple(vert)]
        normal = normals[i]

        if (
          areas[idx] == 0 
          or (idx in branch_pts) 
          or (repair_contacts and contacts[idx] > 0 and not visited[idx])
        ):
          visited[idx] = True
          areas[idx], contact = xs3d.cross_sectional_area(
            binimg, vert, 
            normal, anisotropy,
            return_contact=True,
            use_persistent_data=True,
            segid=segid,
          )
          if repair_contacts:
            contacts[idx] = contact
          else:
            contacts[idx] |= contact # accumulate for branch points
          if idx in branch_pts:
            branch_pt_vals[idx].append(areas[idx])
          if visualize_section_planes:
            img = xs3d.cross_section(
              binimg, vert, 
              normal, anisotropy,
            )
            cross_sections[img > 0] = idx

    if visualize_section_planes:
      import microviewer
      microviewer.view(cross_sections, seg=True)

    for idx, vals in branch_pt_vals.items():
      areas[idx] = sum(vals) / len(vals)

    skel.add_vertex_attribute("cross_sectional_area", areas)
    skel.add_vertex_attribute("cross_sectional_area_contacts", contacts)

  try:
    xs3d.set_shape(all_labels)

    if isinstance(all_labels, CrackleArray):
      iterator = tqdm(
        all_labels.each(crop=True, labels=list(skeletons.keys())),
        disable=(not progress),
        desc="Cross Section Analysis Paths"
      )
      for label, binimg in iterator:
        slc = Bbox.from_slices(bboxes[label])
        cross_sectional_area_helper(skeletons[label], binimg, slc)
    else:
      uniq = unique(all_labels)
      uniq.intersection_update(set(skeletons.keys()))
      iterator = tqdm(
        uniq,
        disable=(not progress),
        desc="Cross Section Analysis Paths"
      )
      roi = Bbox([0,0,0], all_labels.shape[:3])
      for label in iterator:
        cross_sectional_area_helper(skeletons[label], all_labels, roi, label)
  finally:
    xs3d.clear_shape()

  if hasattr(skeletons, "vertices"):
    skelitr = [ skeletons ]
  elif isinstance(skeletons, dict):
    skelitr = skeletons.values()
  else:
    skelitr = iter(skeletons)

  for skel in skelitr:
    if not skel.has("cross_sectional_area"):
      xs_area = np.full(len(skel.vertices), -1, dtype=np.float32, order="F")
      skel.add_vertex_attribute("cross_sectional_area", xs_area)
    if not skel.has("cross_sectional_area_contacts"):
      xs_area = np.full(len(skel.vertices), -1, dtype=np.uint8, order="F")
      skel.add_vertex_attribute("cross_sectional_area_contacts", xs_area)

  return skeletons