IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_OverrideCategoryExtnHistory]'))
DROP TRIGGER [dbo].[tr_OverrideCategoryExtnHistory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_OverrideCategoryExtnHistory] ON [dbo].[OverrideCategoryExtnHistory] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info			Varchar(25),
			@SeqNo			Int,
			@Max			Int,
			@Min			Int,
			@iEmpId			Varchar(10),
			@iComment		Varchar(300),
			@iLastUser		Int,
			@iLastUpdate	Char(10),
			@iArchiveDate	Char(10),
			@cEmpId			Varchar(10),
			@cComment		Varchar(300),
			@cLastUser		Int,
			@cLastUpdate	Char(10),
			@cArchiveDate	Char(10)
			
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(OC.EmpId) 
	From OverrideCategoryExtnHistory OC INNER JOIN deleted DEL 
	ON OC.EmpId = DEL.EmpId) > 0

	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Update'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@iEmpId			= D.EmpId,
				@iComment		= D.Comment,
				@iLastUser		= D.LastUser,
				@iLastUpdate	= D.LastUpdate,
				@iArchiveDate	= D.ArchiveDate
		From	OverrideCategoryExtnHistory O INNER JOIN deleted D 
				ON O.EmpId = D.EmpId
	
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@cEmpId			= O.EmpId,
				@cComment		= O.Comment,
				@cLastUser		= O.LastUser,
				@cLastUpdate	= O.LastUpdate,
				@cArchiveDate	= O.ArchiveDate
		From	OverrideCategoryExtnHistory O INNER JOIN deleted D 
				ON O.EmpId = D.EmpId
				
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		IF	(
			(@cEmpId = @iEmpId) And (@cComment = @iComment) And (@cLastUser = @iLastUser) And
			(@cLastUpdate = @iLastUpdate) And (@cArchiveDate = @iArchiveDate)
			)
		 Begin
			/*********************************************************/
			/*					No Change(s)                         */
			/*********************************************************/
			RETURN
			
		 End
		Else
		 Begin
			/*********************************************************/
			/*				Update Change(s)                         */
			/*********************************************************/
			Insert	AuditOverrideCategoryExtnHistory
			SELECT	D.EmpId,
					D.Comment,
					D.LastUser,
					D.LastUpdate,
					D.ArchiveDate,
					@Info,
					GETDATE(),
					USER_NAME()
			From	OverrideCategoryExtnHistory O INNER JOIN deleted D 
					ON O.EmpId = D.EmpId
												
		 End
		
	End
 Else
	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Delete'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	AuditOverrideCategoryExtnHistory
		SELECT	EmpId,
				Comment,
				LastUser,
				LastUpdate,
				ArchiveDate,
				@Info,
				GETDATE(),
				USER_NAME()
		From	Deleted
		
	End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF
GO