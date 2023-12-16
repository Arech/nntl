function b=isLayerTypeCompound(t)
%by compound we mean here a layer that can contain more than one other layer (Custom layers are excluded!)
assert(t~=0);
LT=static_layer_types();
b = (t==LT.lph || t==LT.lpv);
end