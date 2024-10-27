`ifndef GUARD_INTERFACE
`define GUARD_INTERFACE
interface Stimuli ();
logic [0:3] reqx_cmd ; //bignedian
logic [0:3] reqx_d1;
logic [0:3] reqx_d2;
logic [0:3] reqx_r1;
logic [0:1] reqx_tag;
logic [0:31] reqx_data ;
endinterface 

interface GlobalPorts(input bit clock);
bit reset;
endinterface

interface Output();
logic [0:1] outx_tag;
logic [0:1] outx_resp;
logic [0:31] outx_data;
endinterface
`endif