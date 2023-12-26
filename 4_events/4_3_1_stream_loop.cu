for (int i = 0; i < nStreams; i++) {
int offset = i * bytesPerStream;
cudaMemcpyAsync(&d_a[offset], &a[offset], bytePerStream, streams[i]);
kernel<<grid, block, 0, streams[i]>>(&d_a[offset]);
cudaMemcpyAsync(&a[offset], &d_a[offset], bytesPerStream, streams[i]);
}
for (int i = 0; i < nStreams; i++) {
cudaStreamSynchronize(streams[i]);
}