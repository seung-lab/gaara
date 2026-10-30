import click
import crackle
import numpy as np

import os
import time

class Tuple3(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple3'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
      if len(value) != 3:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
    return value

@click.group()
def main():
	pass

@main.command("fill")
@click.argument("path", type=str)
@click.option('-o', '--output', default=None, help="Path to output filled file.")
@click.option('-v', '--verbose', is_flag=True, default=False, help="Show progress messages.")
@click.option('--no-mmap', is_flag=True, default=False, help="Disallow reading the input file with mmap. This can be important on e.g. NFS mounts.", show_default=True)
@click.option('-t', '--threads', default=0, help="Number of threads to use.", show_default=True)
def fill_holes(path, output, verbose, no_mmap, threads):
	"""Perform morphological hole filling. The thinning algorithm does not work with cavities present."""
	import fastmorph

	arr = crackle.aload(path, allow_mmap=(not no_mmap))
	
	s = time.monotonic()
	filled, holes = fastmorph.fill_holes_v2(
		arr,
		return_crackle=True,
	)
	del holes
	e = time.monotonic()

	if verbose:
		print(f"Filled holes in {e-s:.3f} sec.")

	filled.parallel = threads

	if output is None:
		stem, ext = os.path.splitext(path)
		output = stem + "_filled" + ext

	filled.save(output)
	print(f"wrote: {output}")

@main.command("thin")
@click.argument("path", type=str)
@click.option('-o', '--output', default=None, help="Path to output thinned file.")
@click.option('-v', '--verbose', default=0, help="Show progress messages. 0=off, 1=progress bar, 2=timings, 3=detailed timings")
@click.option('--no-mmap', is_flag=True, default=False, help="Disallow reading the input file with mmap. This can be important on e.g. NFS mounts.", show_default=True)
@click.option('-t', '--threads', default=0, help="Number of threads to use.", show_default=True)
@click.option('--checkpoint', default=0, help="Write checkpoint files every N iterations. (0=off)", show_default=True)
@click.option('--padding', default=6, help="How much padding to use. Must be a multiple of 6.", show_default=True)
@click.option('--memory', type=int, default=-1, help="How much memory in bytes to use.", show_default=True)
def thin(path, output, verbose, no_mmap, threads, checkpoint, padding, memory):
	"""Perform voxel thinning."""
	import gaara

	arr = crackle.aload(path, allow_mmap=(not no_mmap))

	s = time.perf_counter()
	arr = gaara.thin_crackle(
		arr, 
		binary_image=(arr.num_labels() <= 2),
		threads=threads, 
		memory=int(memory), 
		padding=padding,
		verbose=verbose,
		fill_holes=False,
		checkpoints=checkpoint,
	)
	e = time.perf_counter()
	if verbose:
		print(f"Thinned in {e - s:.3f}s")

	if output is None:
		stem, ext = os.path.splitext(path)
		output = stem + "_thinned" + ext

	print(f"wrote: {output}")
	arr.save(output)
	
@main.command("extract")
@click.argument("path", type=str)
@click.option('-o', '--output', default="gaara_skeletons", help="Directory to output skeletons into.")
@click.option('-v', '--verbose', is_flag=True, default=False, help="Show progress messages.")
@click.option('--no-mmap', is_flag=True, default=False, help="Disallow reading the input file with mmap. This can be important on e.g. NFS mounts.", show_default=True)
@click.option('-t', '--threads', default=0, help="Number of threads to use.", show_default=True)
@click.option('--memory', type=int, default=-1, help="How much memory in bytes to use.", show_default=True)
@click.option('--resolution', type=Tuple3(), default="1,1,1", help="How much memory in bytes to use.", show_default=True)
def extract(path, output, verbose, no_mmap, threads, memory, resolution):
	"""Extract skeletons from a thinned image."""
	import gaara

	os.makedirs(output, exist_ok=True)

	arr = crackle.aload(path, allow_mmap=(not no_mmap))

	s = time.perf_counter()
	skeletons = gaara.extract_skeletons_crackle(
		arr,
		memory=int(memory),
		threads=threads,
    verbose=verbose,
	)
	e = time.perf_counter()
	if verbose:
		print(f"Extracted in {e - s:.3f}s")

	provenance = os.path.join(output, "provenance.txt")
	with open(provenance, "wt") as f:
		f.write(f"These skeletons were generated from {path}")

	transform = np.eye(4, dtype=np.float32)
	transform[0,0] = resolution[0]
	transform[1,1] = resolution[1]
	transform[2,2] = resolution[2]
	transform = transform[:3,:]

	for segid, skeleton in skeletons.items():
		skeleton.transform = transform
		save_path = os.path.join(output, f"{segid}.ostd")
		skeleton.save(save_path)
		if verbose:
			print(f"wrote: {save_path}")

@main.command("skeletonize")
@click.argument("path", type=str)
@click.option('-o', '--output', default="gaara_skeletons", help="Directory to output skeletons into.")
@click.option('-v', '--verbose', default=0, help="Show progress messages. 0=off, 1=progress bar, 2=timings, 3=detailed timings")
@click.option('--no-mmap', is_flag=True, default=False, help="Disallow reading the input file with mmap. This can be important on e.g. NFS mounts.", show_default=True)
@click.option('-t', '--threads', default=0, help="Number of threads to use.", show_default=True)
@click.option('--memory', type=int, default=-1, help="How much memory in bytes to use.", show_default=True)
@click.option('--resolution', type=Tuple3(), default="1,1,1", help="How much memory in bytes to use.", show_default=True)
@click.option('--radius/--no-radius', is_flag=True, default=True, help="Compute the distance to boundary.", show_default=True)
@click.option('--padding', default=6, help="How much padding to use. Must be a multiple of 6.", show_default=True)
def skeletonize(path, output, verbose, no_mmap, threads, memory, resolution, radius, padding):
	"""Extract skeletons from a thinned image."""
	import gaara

	os.makedirs(output, exist_ok=True)

	arr = crackle.aload(path, allow_mmap=(not no_mmap))

	s = time.perf_counter()
	skeletons = gaara.skeletonize_crackle(
		arr,
		memory=int(memory),
		threads=threads,
		verbose=verbose,
		radius=radius,
		padding=padding,
	)
	e = time.perf_counter()
	if verbose:
		print(f"Extracted in {e - s:.3f}s")

	provenance = os.path.join(output, "provenance.txt")
	with open(provenance, "wt") as f:
		f.write(f"These skeletons were generated from {path}")

	transform = np.eye(4, dtype=np.float32)
	transform[0,0] = resolution[0]
	transform[1,1] = resolution[1]
	transform[2,2] = resolution[2]
	transform = transform[:3,:]

	for segid, skeleton in skeletons.items():
		skeleton.transform = transform
		save_path = os.path.join(output, f"{segid}.ostd")
		skeleton.save(save_path)
		if verbose:
			print(f"wrote: {save_path}")


