`timescale 1ns/1ns
`include "Interface.sv"
`include "Generator.sv"
module TestBench();
bit clk;
bit a_clock;
bit b_clock;
bit scan_in;
bit scan_out;
initial begin 
clk=0;
a_clock=0;
b_clock=0;
scan_in=0;
scan_out=0;
end
always #10 clk=~clk;
Stimuli t_Port1();
Stimuli t_Port2();
Stimuli t_Port3();
Stimuli t_Port4();
GlobalPorts t_globalPorts(clk);
Output t_outputPort1();
Output t_outputPort2();
Output t_outputPort3();
Output t_outputPort4();

TopProgram topProgram(t_globalPorts,
t_Port1,t_Port2,t_Port3,t_Port4,t_outputPort1,t_outputPort2,
t_outputPort3,t_outputPort4);

calc3_top DUT(
.a_clk(clk),
     .b_clk(clk),
        .c_clk(clk),
        .out1_resp(t_outputPort1.outx_resp),
		.out1_tag(t_outputPort1.outx_tag),
		.out1_data(t_outputPort1.outx_data),
		.req1_cmd(t_Port1.reqx_cmd),
		.req1_data(t_Port1.reqx_data),
		.req1_d1(t_Port1.reqx_d1),
		.req1_d2(t_Port1.reqx_d2),
		.req1_r1(t_Port1.reqx_r1),
		.req1_tag(t_Port1.reqx_tag),
		
		.out2_resp(t_outputPort2.outx_resp),
		.out2_tag(t_outputPort2.outx_tag),
		.out2_data(t_outputPort2.outx_data),
		.req2_cmd(t_Port2.reqx_cmd),
		.req2_data(t_Port2.reqx_data),
		.req2_d1(t_Port2.reqx_d1),
		.req2_d2(t_Port2.reqx_d2),
		.req2_r1(t_Port2.reqx_r1),
		.req2_tag(t_Port2.reqx_tag),
		
		.out3_resp(t_outputPort3.outx_resp),
		.out3_tag(t_outputPort3.outx_tag),
		.out3_data(t_outputPort3.outx_data),
		.req3_cmd(t_Port3.reqx_cmd),
		.req3_data(t_Port3.reqx_data),
		.req3_d1(t_Port3.reqx_d1),
		.req3_d2(t_Port3.reqx_d2),
		.req3_r1(t_Port3.reqx_r1),
		.req3_tag(t_Port3.reqx_tag),
		
		.out4_resp(t_outputPort4.outx_resp),
		.out4_tag(t_outputPort4.outx_tag),
		.out4_data(t_outputPort4.outx_data),
		.req4_cmd(t_Port4.reqx_cmd),
		.req4_data(t_Port4.reqx_data),
		.req4_d1(t_Port4.reqx_d1),
		.req4_d2(t_Port4.reqx_d2),
		.req4_r1(t_Port4.reqx_r1),
		.req4_tag(t_Port4.reqx_tag),
		
        
		
        .scan_in(scan_in),
        .scan_out(scan_out),
        .reset(t_globalPorts.reset)

);
endmodule