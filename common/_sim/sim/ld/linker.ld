INCLUDE output_format.ld
ENTRY(_start)

__DYNAMIC = 0;

INCLUDE regions.ld


SECTIONS
{
	.text :
	{
		_ftext = .;
		*(.text.start)
		*(.text .stub .text.* .gnu.linkonce.t.*)
		_etext = .;
	} > main_ram
	/*} > rom */

	.rodata :
	{
		. = ALIGN(8);
		_frodata = .;
		*(.rodata .rodata.* .gnu.linkonce.r.*)
		*(.rodata1)
		*(.srodata .srodata.*)
		. = ALIGN(8);
		_erodata = .;
	} > main_ram
	/*} > rom */

	.data : AT (ADDR(.rodata) + SIZEOF (.rodata))
	{
		. = ALIGN(8);
		_fdata = .;
		*(.data .data.* .gnu.linkonce.d.*)
		*(.data1)
		*(.ramtext .ramtext.*)
		_gp = ALIGN(16);
		*(.sdata .sdata.* .gnu.linkonce.s.* .sdata2 .sdata2.*)
		_edata = ALIGN(16); /* Make sure _edata is >= _gp. */
	} > main_ram
	/*} > sram */

	.bss : AT (ADDR(.data) + SIZEOF (.data))
	{
		. = ALIGN(16);
		_fbss = .;
		*(.dynsbss)
		*(.sbss .sbss.* .gnu.linkonce.sb.*)
		*(.scommon)
		*(.dynbss)
		*(.bss .bss.* .gnu.linkonce.b.*)
		*(COMMON)
		. = ALIGN(8);
		_ebss = .;
		_end = .;
		__bss_end = .;
	} > main_ram


	/*} > sram */
        .arena (NOLOAD) :
        {
                _farena = .;
                *(.arena)
                _earena = .;
        } > arena

}

PROVIDE(_fstack = ORIGIN(main_ram) + LENGTH(main_ram) - 4);
/* PROVIDE(_fstack = ORIGIN(sram) + LENGTH(sram) - 4); */
PROVIDE( __end = __bss_end );
PROVIDE( _end = __bss_end );
PROVIDE( end = __bss_end );
PROVIDE( __bss_start = ADDR(.bss));
PROVIDE( __bss_size = __bss_end - __bss_start );

/* Make the rest of memory available for heap storage */
PROVIDE (__heap_start = __end);
PROVIDE (__heap_end = _fstack - (DEFINED(__stack_size) ? __stack_size : 0x800));
PROVIDE (__heap_size = __heap_end - __heap_start);
