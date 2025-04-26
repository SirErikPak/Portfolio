if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[P_PfizerDemographic]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[P_PfizerDemographic]
GO
Create Proc P_PfizerDemographic
AS
/*********************************************************************/
/*       WYETH Demographic – Create Demographic File                 */
/*********************************************************************/
/*                                                                   */
/*                                                                   */
/*                                                                   */
/*                ---Date---  ---name---  --Purpose----------------- */
/*  Created       10/21/2009  EPAK                                   */
/*  Mod 01          /  /       .                                     */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*								     */
/*********************************************************************/
Declare		@ErrorCode	Int,
		@Zero		Int,
		@Error		Char(1),
		@Yes		Char(1),
		@No		Char(1),
		@YYYYMMDD	Char(8),
		@Step		Varchar(128),
		@Process	Varchar(128),
		@XlsName	Varchar(255),
		@XLS		Char(4),
		@INP		Char(4),
		@CopyCmd	Varchar(1024),
		@RenameCmd	Varchar(1024),
		@ArchiveCmd	Varchar(1024),
		@IsofCmd	Varchar(1024),
		@SourcePath	Varchar(128),
		@TargetPath	Varchar(128),
		@MainSourcePath	Varchar(128)

/*********************************************************************/
/*								     */
/*********************************************************************/
Select	@Zero		= 0,
	@Yes		= 'Y',
	@No		= 'N',
	@Process	= 'P_PfizerDemographic',
	@XlsName	= 'WYETH-PROD-MADISON-EquityDemographics',
	@XLS		= '.xls',
	@INP		= '.inp',
	@YYYYMMDD	= (CONVERT(Char(8),Getdate(),112)),
	@MainSourcePath	= 'E:\Data\LTIDoc\Demographics\',
--	@TargetPath	= '\\GVWMDVM018\Isoft\outboxSTOPTS\',
	@TargetPath	= '\\MD02N49\Data\LTIDoc\',
	@SourcePath	= 'E:\Data\LTIDoc\Demographics\Archive\'


/*********************************************************************/
/*								     */
/*********************************************************************/
SET	@Error		= @Yes
SET	@ArchiveCmd	= 'Copy ' + @SourcePath + @XlsName + @XLS + ' ' + @SourcePath + @XlsName + '_' + @YYYYMMDD + @XLS
SET	@IsofCmd	= 'Copy ' + @SourcePath + @XlsName + @XLS + ' ' + @TargetPath + @XlsName + @INP
SET	@RenameCmd	= 'Ren '  + @TargetPath + @XlsName + @INP + ' ' + @XlsName + '+' + @YYYYMMDD + @XLS
SET	@CopyCmd	= 'Copy ' + @MainSourcePath + @XlsName + @XLS + ' ' + @SourcePath + @XlsName + @XLS

/*********************************************************************/
/*		Backup With Date (YYYMMDD)			     */
/*********************************************************************/
SET	@Step	= '01 - Archive (' + @ArchiveCmd + ')'

EXEC @ErrorCode = Master..xp_cmdshell @ArchiveCmd, no_output

		IF @ErrorCode <> @Zero
			GOTO   SucessError


/*********************************************************************/
/*		Copy to ISOF with IMP Extension			     */
/*********************************************************************/
SET	@Step	= '02 - ISOF (' + @IsofCmd + ')'

EXEC @ErrorCode = Master..xp_cmdshell @IsofCmd, no_output

		IF @ErrorCode <> @Zero
			GOTO   SucessError

/*********************************************************************/
/*		ISOF Rename Ext to XLS				     */
/*********************************************************************/
SET	@Step	= '03 - Rename (' + @RenameCmd + ')'

EXEC @ErrorCode = Master..xp_cmdshell @RenameCmd, no_output

		IF @ErrorCode <> @Zero
			GOTO   SucessError

/*********************************************************************/
/*		Copy Fresh XLS into Working Archive Folder	     */
/*********************************************************************/
SET	@Step	= '04 - Copy (' + @CopyCmd + ')'

EXEC @ErrorCode = Master..xp_cmdshell @CopyCmd, no_output

		IF @ErrorCode <> @Zero
			GOTO   SucessError


/*********************************************************************/
SET	@Error = @No

/*********************************************************************/
/*								     */
/*********************************************************************/
SucessError:
IF	@Error = @Yes
 Begin
	INSERT	WyethAux.dbo.ProcessLog
	VALUES 	(Getdate(), @Process,@ErrorCode,'FAILURE',@Process + ' FAILED AT STEP : '+ @Step)
	RAISERROR (56001,16,1)   -- Custom SQL error-message in MASTER
	RETURN	@ErrorCode

 End
ELSE
 Begin
	INSERT 	WyethAux.dbo.ProcessLog
	VALUES	(Getdate(), @Process,@ErrorCode,'SUCCESS',@Process + ' Completed')
 End

/*********************************************************************/
/*								     */
/*********************************************************************/
SET NOCOUNT OFF